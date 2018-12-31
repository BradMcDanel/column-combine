from __future__ import print_function
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import torch.nn.functional as F
from torch.autograd import Variable

import net
import time
import gc
import os

def build_model(args):
    settings = list(zip(args.filters, args.layers, args.strides, args.groups))
    if args.layer_type == 'vgg':
        layer = net.make_vgg_layer(args)
    else:
        layer = net.make_shift_layer(args)
    model = net.ShiftMobile(settings, layer=layer,
                            in_channels=3*(args.reshape_stride**2),
                            n_class=args.n_class, dropout=args.dropout)
    return model


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decrease = args.epochs // 3
    lr = args.lr * (0.1 ** (epoch // decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda is not None:
            x = x.cuda()
        target = target.cuda()

        # compute output
        output = data_parallel(model, x)
        loss = criterion(output, target)
        if args.l1_penalty > 0:
            loss += args.l1_penalty*l1_weight_total(model)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # record stats in model for visualization
        model.stats['train_loss'].append(loss.item())

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Train:: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader) - 1, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, target) in enumerate(val_loader):
            if args.cuda is not None:
                x = x.cuda()
            target = target.cuda()

            # compute output
            output = data_parallel(model, x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # record stats in model for visualization

        print('Test :: [{0}][{1}/{2}]\t'
              'Loss {loss.avg:.4f}\t'
              'Acc@1 {top1.avg:.3f}\t'
              'Acc@5 {top5.avg:.3f}'.format(
              epoch, i, len(val_loader) - 1,
              loss=losses, top1=top1, top5=top5))

    model.stats['test_loss'].append(losses.avg)
    model.stats['test_acc'].append(top1.avg)
    return losses.avg, top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def l1_weight_total(model):
    l1_total = 0
    for layer in get_conv_layers(model):
        l1_total += layer._weight.norm(1)
    return l1_total

def num_nonzeros(model, top=True):
    '''
    Only considers conv layers for now
    '''
    if model == None:
        return 0

    non_zeros, total = 0, 0
    for layer in model.children():
        if  isinstance(layer, net.Conv2d):
            flat_w = layer.mask.data.cpu().numpy().flatten()
            non_zeros += np.sum(flat_w != 0)
            total += len(flat_w)
        elif isinstance(layer, nn.Conv2d):
            if not layer.weight.requires_grad:
                continue
            B, C, W, H = layer.weight.shape
            total_W = B*C*W*H
            non_zeros += total_W
            total += total_W
        else:
            n, t = num_nonzeros(layer, False)
            non_zeros += n
            total += t
    
    return int(non_zeros), int(total)

def prune(model, prune_progress):
    model.cpu()
    layers = get_conv_layers(model)
    for layer_idx, layer in enumerate(layers):
        prune_pct = prune_progress * (1 - (1 / layer.groups))
        weight = layer._weight.data.abs().view(-1)
        num_weights = len(weight)
        num_prune = math.ceil(num_weights * prune_pct)
        prune_idxs = weight.sort()[1][:num_prune]
        mask = torch.ones(num_weights)
        mask[prune_idxs] = 0
        layer._weight.data[prune_idxs] = 0
        layer._mask = mask
    
    model.cuda()


def prune_group(model, prune_progress):
    model.cpu()
    layers = get_conv_layers(model)
    for layer_idx, layer in enumerate(layers):
        prune_pct = prune_progress * (1 - (1 / layer.groups))
        weight = layer._mask * layer._weight
        weight = weight.data.abs().view(-1, layer.groups)

        # at least one entry per prune group must survive
        max_w = weight.max(1)[1]
        max_w += layer.groups*torch.arange(len(max_w))
        weight = weight.view(-1)
        weight[max_w] += 1e8

        num_weights = len(weight)
        num_prune = math.ceil(num_weights * prune_pct)
        prune_idxs = weight.sort()[1][:num_prune]
        mask = torch.ones(num_weights)
        mask[prune_idxs] = 0
        layer._weight.data[prune_idxs] = 0
        layer._mask = mask
    
    model.cuda()

def target_nonzeros(model):
    layers = get_conv_layers(model)
    total_weights = 0
    for layer_idx, layer in enumerate(layers):
        num_weights = len(layer._weight)
        total_weights +=  (1 / layer.groups) * num_weights
    
    return total_weights

def get_max_weight(model):
    max_w = -1e10
    for layer in get_conv_layers(model):
        max_w = max(max_w, layer.weight.abs().max().item())
    return max_w

def get_weights(model, float_weight=False):
    weights = []
    layers = get_conv_layers(model)
    for layer in layers:
        if float_weight:
            weights.extend(layer._weight.view(-1).data.cpu().tolist())
        else:
            weights.extend(layer.weight.view(-1).data.cpu().tolist())
    return np.array(weights)

def get_nonzero_layer_size(model):
    sizes = []
    layers = get_conv_layers(model)
    for i, layer in enumerate(layers):
        w = layer.weight.data.cpu().numpy()
        B, C, W, H = w.shape
        w = w.reshape(B, C*W*H)
        r, c = (w.sum(1) != 0).sum(), (w.sum(0) != 0).sum()
        if i > 0:
            c = min(c, sizes[i-1][0])
            sizes[i-1][0] = c
        sizes.append([r, c])
    
    return sizes

def get_nonzero_layer_ratio(model):
    ratios = []
    layers = get_conv_layers(model)
    for layer in layers:
        w = layer.weight.data.cpu().numpy().flatten()
        ratios.append((w != 0).sum() / float(len(w)))
    return ratios

def get_conv_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.Conv2d):
            layers.append(layer)
        else:
            layers.extend(get_conv_layers(layer))

    return layers

def set_batchnorm_alpha(model, alpha):
    for layer in get_batchnorm_layers(model):
        layer.alpha = alpha

def get_batchnorm_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, nn.BatchNorm2d):
            layers.append(layer)
        else:
            layers.extend(get_batchnorm_layers(layer))

    return layers

def get_shift_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.Shift):
            layers.append(layer)
        else:
            layers.extend(get_shift_layers(layer))

    return layers