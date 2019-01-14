from __future__ import print_function

import argparse
import os
from pprint import pprint
import multiprocessing 

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
cudnn.benchmark = True 

import datasets
import util
import packing


def train(model, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    prune_epoch = 0
    max_prune_rate = 0.8
    final_prune_epoch = int(0.5*args.epochs)
    num_prune_epochs = 10
    prune_rates = [max_prune_rate*(1 - (1 - (i / num_prune_epochs))**3)
                   for i in range(num_prune_epochs)]
    prune_rates[-1] = max_prune_rate
    prune_epochs = np.linspace(0, final_prune_epoch, num_prune_epochs).astype('i').tolist()
    print("Pruning Epochs: {}".format(prune_epochs))
    print("Pruning Rates: {}".format(prune_rates))

    curr_weights, num_weights = util.num_nonzeros(model)
    macs = curr_weights

    model.stats = {'train_loss': [], 'test_acc': [], 'test_loss': [],
                    'weight': [], 'lr': [], 'macs': [], 'efficiency': []}
    best_path = args.save_path.split('.pth')[0] + '.best.pth'
    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        for g in optimizer.param_groups:     
            lr = g['lr']                    
            break        

        # prune smallest weights up to a set prune_rate
        if epoch in prune_epochs:
            util.prune(model, prune_rates[prune_epoch])
            curr_weights, num_weights = util.num_nonzeros(model)
            prune_epoch += 1

        # # final pruning stage (perform column combining)
            packing.pack_model(model, args.gamma)
            macs = np.sum([x*y for x, y in model.packed_layer_size])
            curr_weights, num_weights = util.num_nonzeros(model)

        if epoch == prune_epochs[-1]:
            # disable l1 penalty, as target sparsity is reached
            args.l1_penalty = 0

        print('     :: [{}]\tLR {:.4f}\tNonzeros ({}/{})'.format(
            epoch, lr, curr_weights, num_weights))
        train_loss = util.train(train_loader, model, criterion, optimizer, epoch, args)
        test_loss, test_acc = util.validate(val_loader, model, criterion, epoch, args)

        is_best = test_acc > best_test_acc
        best_test_acc = max(test_acc, best_test_acc)
        model.stats['lr'].append(lr)
        model.stats['macs'].append(macs)
        model.stats['weight'].append(curr_weights)
        model.stats['efficiency'].append(100.0 * (curr_weights / macs))
        model.optimizer = optimizer.state_dict()
        model.epoch = epoch

        model.cpu()
        torch.save(model, args.save_path)
        if is_best:
            torch.save(model, best_path)
        model.cuda()
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Training Script')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--input-size', type=int, help='spatial width/height of input')
    parser.add_argument('--n-class', type=int, help='number of classes')
    parser.add_argument('--aug', default='+', help='data augmentation level (`-`, `+`)')
    parser.add_argument('--save-path', required=True, help='path to save model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--sample-ratio', type=float, default=1.,
                        help='ratio of training data used')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--l1-penalty', type=float, default=0.0,
                        help='l1 penalty (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='column combine gamma parameter (default: 0.5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='printing frequency')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load-path', default=None,
                        help='path to load model - trains new model if None')
    parser.add_argument('--reshape-stride', type=int, default=1, help='checkerboard reshape stride')
    parser.add_argument('--filters', nargs='+',  type=int, help='size of layers in each block')
    parser.add_argument('--layers', nargs='+', type=int, help='number of layers for each block')
    parser.add_argument('--strides', nargs='+', type=int, help='stride for each block')
    parser.add_argument('--groups', nargs='+', type=int, help='number of sparse groups')
    parser.add_argument('--layer-type', default='shift', choices=['vgg', 'shift'],
                        help='type of layer')
    parser.add_argument('--dropout', action='store_true')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Arguments:')
    pprint(args.__dict__, width=1)

    #set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                                args.cuda, args.aug, input_size=args.input_size,
                                sample_ratio=args.sample_ratio)
    train_dataset, train_loader, test_dataset, test_loader = data

    # load or create model
    if args.load_path == None:
        model = util.build_model(args)
    else:
        model = torch.load(args.load_path)

    if args.cuda:
        model = model.cuda()

    print(model)
    print(util.num_nonzeros(model))
    print('Target Nonzeros:', util.target_nonzeros(model))

    train(model, train_loader, test_loader, args)
