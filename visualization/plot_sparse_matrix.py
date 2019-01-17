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

from packing import densify
import datasets
import net
import util
import matplotlib
import matplotlib.pyplot as plt
from config import init_mpl_settings
init_mpl_settings()


model_path = 'models/cifar10/cifar10-shift-1.75.pth'
layer_idx = 7
arr_x, arr_y = 64, 64
model = torch.load(model_path)

layer = model[layer_idx][1].weight.data.cpu().numpy()
B, C, _, _ = layer.shape
layer = layer.reshape(B, C)
rows, cols = model.packed_layer_idxs[layer_idx]
layer_packed = densify(layer, cols, rows)
layer_packed[layer_packed != 0] = 1

# sort for display purposes 
idxs = layer_packed.sum(0).argsort()
layer_packed = layer_packed[:, idxs][:, ::-1]

layer = layer[layer.sum(1) != 0]
layer = layer[:, layer.sum(0) != 0]
layer[layer != 0] = 1

im = plt.imshow(layer, interpolation='none', cmap='Greys', vmin=0, vmax=1)
ax = plt.gca()
ax.set_xticks(np.arange(arr_y, layer.shape[1], arr_y))
ax.set_yticks(np.arange(arr_x, layer.shape[0], arr_x))
ax.xaxis.tick_top()
ax.grid(which='major', color='red', linestyle='-', linewidth=2)
plt.tight_layout()
plt.savefig('visualization/figures/layer_sparse.png', dpi=300)
plt.clf()


im = plt.imshow(layer_packed, interpolation='none', cmap='Greys', vmin=0, vmax=1)
ax = plt.gca()
ax.set_xticks(np.arange(arr_y, layer_packed.shape[1], arr_y))
ax.set_yticks(np.arange(arr_x, layer_packed.shape[0], arr_x))
ax.xaxis.tick_top()
ax.grid(which='major', color='red', linestyle='-', linewidth=2)
plt.tight_layout()
plt.savefig('visualization/figures/layer_combine.png', dpi=300)
plt.clf()