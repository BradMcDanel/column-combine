'''
Modified Version of: https://github.com/aaron-xichen/pytorch-playground/blob/master/quantize.py 
'''

import copy
from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math

import net

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
    sf = math.ceil(math.log(v+1e-12,2))
    return sf

def linear_quantize(x, delta, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(x) - 1
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(x / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits)
    v = torch.exp(v) * s
    return v

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy()[0])
        min_val = float(min_val.data.cpu().numpy()[0])

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v


class LinearQuant(nn.Module):
    def __init__(self, bits, delta):
        super(LinearQuant, self).__init__()
        self.bits = bits
        self.delta = delta

    def forward(self, x):
        output = linear_quantize(x, self.delta, self.bits)
        # print(output.unique().shape)
        return output

    def extra_repr(self):
        return 'delta={}, bits={}'.format(self.delta, self.bits)

class ShiftLinearQuant(nn.Module):
    def __init__(self, name, bits, counter=50):
        super(ShiftLinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.bound = None

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            bounds = torch.Tensor([6 / 2**(self.bits-i-1) for i in range(self.bits)]).cuda().float()
            bound_new = bounds[(bounds - input.max() < 0).nonzero()[-1]+1].item()
            self.bound = min(self.bound, bound_new) if self.bound is not None else bound_new
            return input
        else:
            integer_input = torch.round(input / self.bound * (math.pow(2, self.bits) - 1))
            output = integer_input / (math.pow(2, self.bits) - 1) * self.bound
            return output

    def __repr__(self):
        return '{}(bound={}, bits={}, counter={})'.format(
            self.__class__.__name__, self.bound, self.bits, self.counter)

class Tracker(nn.Module):
    def __init__(self, layer):
        super(Tracker, self).__init__()
        self.layer = layer
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        h = self.layer(x)
        self.output = h
        return h

def quantize_weight(input, bits=8):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    # find the max of the input
    max_value = input.abs().view(-1).max()
    print(max_value.item())
    integer_input = torch.round(input/max_value * (math.pow(2,bits-1)-1))
    quantized_output = integer_input/(math.pow(2,bits-1)-1) * max_value
    return quantized_output


class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)


def apply_mask(model):
    '''
    Applies mask to internal _weight. Required for quantization.
    '''
    for layer in model.children():
        if isinstance(layer, net.MConv2d):
            layer._weight.data = layer._weight.data * layer.mask.data
        else:
            apply_mask(layer)

def quantize_model(model, param_bits, act_bits=None, bn_bits=10, overflow_rate=0.0, counter=10):
    model = copy.deepcopy(model)
    apply_mask(model)
    if act_bits is None:
        act_bits = param_bits

    state_dict = model.state_dict()
    state_dict_quant = OrderedDict()
    sf_dict = OrderedDict()
    for k, v in state_dict.items():
        if '_mask' in  k:
            state_dict_quant[k] = v
            continue
        elif 'running' in k:
            bits = bn_bits
        else:
            bits = param_bits

        print(k)
        v_quant = quantize_weight(v, bits)
        state_dict_quant[k] = v_quant
    model.load_state_dict(state_dict_quant)

    return quantize_model_internal(model, act_bits, overflow_rate, counter)

def quantize_model_internal(model, bits, overflow_rate=0.0, counter=10):
    for k, layer in model._modules.items():
        if isinstance(layer, nn.ReLU6):
            model._modules[k] = Tracker(nn.Sequential(
                layer,
                ShiftLinearQuant('{}_quant'.format(k), bits=bits, counter=counter),
            ))
        elif isinstance(layer, net.MConv2d):
            model._modules[k] = Tracker(layer)
        elif isinstance(layer, net.Shift):
            model._modules[k] = Tracker(layer)
        elif isinstance(layer, nn.MaxPool2d):
            model._modules[k] = Tracker(layer)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            model._modules[k] = Tracker(layer)
        else:
            quantize_model_internal(layer, bits, overflow_rate, counter)
    
    return model