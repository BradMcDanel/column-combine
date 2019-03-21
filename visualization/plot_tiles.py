from __future__ import print_function

import argparse
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tqdm import tqdm
import math

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import datasets
import net
import util
import matplotlib
import matplotlib.pyplot as plt
from config import init_mpl_settings
init_mpl_settings()

arr_x, arr_y = 64, 64
model_paths = ['models/cifar10/cifar10-shift-0.0.pth',
               'models/cifar10/cifar10-shift-0.25.pth',
               'models/cifar10/cifar10-shift-1.75.pth']
separate_tiles, combine_tiles = [], []
for model_path in model_paths:
    model = torch.load(model_path)
    tiles = []
    for x, y in util.get_nonzero_layer_size(model):
        xr, yr = math.ceil(x / arr_x), math.ceil(y / arr_y)
        tiles.append(xr*yr)
    separate_tiles.append(tiles)

    tiles = []
    for x, y in model.packed_layer_size:
        xr, yr = math.ceil(x / arr_x), math.ceil(y / arr_y)
        tiles.append(xr*yr)
    combine_tiles.append(tiles)

tiles = [combine_tiles[0], combine_tiles[1], combine_tiles[2]]
names = [
         r'Unstructured Pruning',
         r'Column-Combine Pruning ($\gamma={}$)'.format(0.25),
         r'Column-Combine Pruning ($\gamma={}$)'.format(1.75),
        ]
colors = ['coral', 'mediumseagreen', 'cornflowerblue']

width = 0.28 
fig, ax = plt.subplots(figsize=(8,6))
ax.yaxis.grid()
ax.set_axisbelow(True)
pos = np.array(range(len(tiles[0])))
for i in range(len(tiles)):
    plt.bar(pos + i*width, tiles[i], width, color=colors[i], label=names[i])

ax.set_xlabel('CIFAR-10 VGG-19 (1x1) Convolution Layer')
ax.set_ylabel('Number of Tiles')
ax.set_ylim(0, 1.05*max(tiles[0]))
ax.set_xlim(-0.4, len(tiles[0])+0.2)
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(np.arange(1, len(tiles[0])+1))
plt.legend(loc=0, framealpha=1.0, handletextpad=0.15)
plt.tight_layout()
plt.savefig('visualization/figures/tiles.pdf', dpi=300)
plt.clf()
