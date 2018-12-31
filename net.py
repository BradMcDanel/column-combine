import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.cpp_extension import load
from torch.distributions import categorical

from itertools import product
import util

shift_cuda = load(
    'shift_cuda', ['kernels/shift_cuda.cpp', 'kernels/shift_cuda_kernel.cu'], extra_cflags=['-O3'])


def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)


class shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift):
        ctx.save_for_backward(shift)
        return shift_cuda.forward(x, shift)

    @staticmethod
    def backward(ctx, grad_output):
        shift, = ctx.saved_tensors
        grad_output = shift_cuda.backward(grad_output, shift)

        return grad_output, None


class Shift(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Shift, self).__init__()
        self.channels = in_channels
        self.kernel_size = kernel_size
        if kernel_size == 3:
            p = torch.Tensor([0.3, 0.4, 0.3])
        elif kernel_size == 5:
            p = torch.Tensor([0.1, 0.25, 0.3, 0.25, 0.1])
        elif kernel_size == 7:
            p = torch.Tensor([0.075, 0.1, 0.175, 0.3, 0.175, 0.1, 0.075])
        elif kernel_size == 9:
            p = torch.Tensor([0.05, 0.075, 0.1, 0.175, 0.2, 0.175, 0.1, 0.075, 0.05])
        else:
            raise RuntimeError('Unsupported kernel size')

        shift_t = categorical.Categorical(p).sample((in_channels, 2)) - (kernel_size // 2)
        self.register_buffer('shift_t', shift_t.int())
    
    def forward(self, x):
        if x.is_cuda:
            return shift.apply(x, self.shift_t)
        else:
            print('Shift only supports GPU for now..')
            assert False

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}')
        return s.format(**self.__dict__)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(Conv2d, self).__init__()
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.dilation = _make_pair(dilation)
        self.groups = groups
        self.bias = None
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        N = out_channels*in_channels*kernel_size*kernel_size
        n = kernel_size * kernel_size * out_channels
        self._weight = nn.Parameter(torch.Tensor(N))
        self._weight.data.normal_(0, math.sqrt(2. / n))

        self.register_buffer('_mask', torch.ones(N))

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
                    
    @property
    def weight(self):
        w = self.mask * self._weight
        return w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

    @property
    def mask(self):
        return Variable(self._mask, requires_grad=False)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class CheckerboardReshape(nn.Module):
    def __init__(self, s):
        super(CheckerboardReshape, self).__init__()
        self.s = s
        self.idxs = list(range(s))
 
    def __call__(self, x):
        B, C, W, H = x.shape
        h = torch.stack([x[:, :, i::self.s, j::self.s] for i, j in product(self.idxs, self.idxs)], 1)
        h = h.reshape(B, -1, W // self.s, H // self.s)
        return h


def make_shift_layer(args):
    def shift_layer(in_channels, out_channels, stride, groups, layer_idx, num_layers):
        layer = []
        first = layer_idx == 0
        last = layer_idx == num_layers - 1

        if first:
            if args.reshape_stride != 1:
                layer.append(CheckerboardReshape(args.reshape_stride))
            layer.append(Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
        elif last:
            layer.append(nn.AdaptiveAvgPool2d(1))
            layer.append(View((-1, in_channels)))
            layer.append(nn.Linear(in_channels, out_channels))
        else:
            layer.append(Shift(in_channels, 3))
            layer.append(Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))

        layer = nn.Sequential(*layer)

        return layer

    return shift_layer

def make_vgg_layer(args):
    def vgg_layer(in_channels, out_channels, stride, groups, layer_idx, num_layers):
        layer = []
        first = layer_idx == 0
        last = layer_idx == num_layers - 1

        if first:
            if args.reshape_stride != 1:
                layer.append(CheckerboardReshape(args.reshape_stride))
            layer.append(Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
        elif last:
            layer.append(nn.AdaptiveAvgPool2d(1))
            layer.append(View((-1, in_channels)))
            layer.append(nn.Linear(in_channels, out_channels))
        else:
            layer.append(Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))

        layer = nn.Sequential(*layer)

        return layer

    return vgg_layer

class ShiftMobile(nn.Module):
    def __init__(self, settings, layer, in_channels, n_class, dropout=False):
        super(ShiftMobile, self).__init__()
        input_channel = settings[0][0]
        layer_idx = 0
        num_layers = sum([n for c, n, s, g in settings]) + 2
        self.num_layers = num_layers
        layers = [layer(in_channels, input_channel, 1, 1, layer_idx, num_layers)]
        layer_idx += 1

        prev_groups = settings[0][3]
        for k, (c, n, s, g) in enumerate(settings):
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                groups = prev_groups if i == 0 else g
                layers.append(layer(input_channel, output_channel, stride,
                                    groups, layer_idx, num_layers))
                input_channel = output_channel
                layer_idx += 1
                prev_groups = g

        if dropout:
            layers.append(nn.Dropout(0.5))

        layers.append(layer(input_channel, n_class, 1, 1, layer_idx, num_layers))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, i):
        return self.model[i]

    def __len__(self):
        return self.num_layers
