from __future__ import print_function

import argparse
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tqdm import tqdm

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

title = 'CIFAR-10 VGG-19 (1x1)'
model_path = 'models/cifar10/cifar10-shift-1.75.pth'
model = torch.load(model_path)

fig, ax1 = plt.subplots()
ax1.set_title(title)
epochs = len(model.stats['test_acc'])
ma = util.running_mean(model.stats['test_acc'], 5)
ln1 = ax1.plot(np.arange(len(ma))+5, ma, 'b-', linewidth=3, label='Classification Accuracy (%)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Classification Accuracy (%)', color='b')
ax2 = ax1.twinx()

# add prune lines
for i in range(1, len(model.stats['weight'])):
    if model.stats['weight'][i] != model.stats['weight'][i-1]:
        ax2.axvline(i, linestyle='--', linewidth=2, color='grey')

ln2 = ax2.plot(model.stats['weight'], '-', color='r', linewidth=3, label='Nonzero Weights')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Nonzero Weights', color='r')

yticks = np.arange(400000, 2000001, 400000)
ax2.set_yticks(yticks)
ax2.set_yticklabels(['{:.1f}M'.format(y / 1000000) for y in yticks])

ax1.set_zorder(ax2.get_zorder()+1)
ax1.patch.set_visible(False)

fig.tight_layout()
plt.savefig('visualization/figures/epoch-acc-weight.pdf', dpi=300)
plt.clf()
