import argparse
import math

import torch
import net
import torch.nn as nn
import datasets
import util
import quant


def quant_linear(linear, bits=8):
    w, b = linear.weight.data, linear.bias.data

    sf = bits - 1. - quant.compute_integral_part(w, 0.0)
    delta = math.pow(2.0, -sf)
    linear.weight.data = quant.linear_quantize(w, delta, bits)

    sf = bits - 1. - quant.compute_integral_part(b, 0.0)
    delta = math.pow(2.0, -sf)
    linear.bias.data = quant.linear_quantize(b, delta, bits)

    return linear

def fuse_conv_bn(conv, bn, bits=8):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.mask = conv.mask

    sf = bits - 1. - quant.compute_integral_part(w, 0.0)
    delta = math.pow(2.0, -sf)
    w = quant.linear_quantize(w, delta, bits)
    fused_conv.weight = nn.Parameter(w)

    sf = bits - 1. - quant.compute_integral_part(b, 0.0)
    delta = math.pow(2.0, -sf)
    b = quant.linear_quantize(b, delta, bits)
    fused_conv.bias = nn.Parameter(b)

    return fused_conv


def fuse_shift_quantize_model(model, bits=8):
    for i, layer in enumerate(model):
        if i == 0:
            fuse_layer = nn.Sequential(
                fuse_conv_bn(layer[0], layer[1]),
                layer[2],
                quant.LinearQuant(bits, 6 / 2**bits)
            )
            model.model[i] = fuse_layer
        elif i < len(model) - 1:
            fuse_layer = nn.Sequential(
                layer[0],
                fuse_conv_bn(layer[1], layer[2]),
                layer[3],
                quant.LinearQuant(bits, 6 / 2**bits)
            )
            model.model[i] = fuse_layer
        elif i == len(model):
            fuse_layer = nn.Sequential(
                layer[0],
                layer[1],
                quant_linear(layer[2]),
            )
            model.model[i] = fuse_layer

def fuse_vgg_quantize_model(model, bits=8):
    for i, layer in enumerate(model):
        if i < len(model) - 1:
            fuse_layer = nn.Sequential(
                fuse_conv_bn(layer[0], layer[1]),
                layer[2],
                quant.LinearQuant(bits, 6 / 2**bits)
            )
            model.model[i] = fuse_layer
        elif i == len(model):
            fuse_layer = nn.Sequential(
                layer[0],
                layer[1],
                quant_linear(layer[2]),
            )
            model.model[i] = fuse_layer
 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Training Script')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--input-size', type=int, help='spatial width/height of input')
    parser.add_argument('--n-class', type=int, help='number of classes')
    parser.add_argument('--aug', default='+', help='data augmentation level (`-`, `+`)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--load-path', help='path to model')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss().cuda()
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                                args.cuda, args.aug, input_size=args.input_size,
                                val_only=True)
    train_dataset, train_loader, test_dataset, test_loader = data
    model = torch.load(args.load_path)

    print('Before Fuse + Quantize: {:2.4f}'.format(model.stats['test_acc'][-1]))

    if type(model[1][0]) == net.Shift:
        fuse_shift_quantize_model(model)
    else:
        fuse_vgg_quantize_model(model)

    model.cuda()
    _, top1 = util.validate(test_loader, model, criterion, 0, args, no_print=True)
    print('After Fuse + Quantize:  {:2.4f}'.format(top1))