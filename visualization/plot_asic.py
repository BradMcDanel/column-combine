from __future__ import print_function

import argparse
import os
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
import measurement
import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 23
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data = [
    {
        'throughput': [1, 1, 1], 
        'power': [1, 1, 1],
        'tiles': [1, 1, 1],
        'acc': [98.5, 91.67, 93.96]
    },
    {
        'throughput': [1.08, 1.031, 1.17],
        'power': [0.6, 0.923, 0.842],
        'tiles': [0.935, 0.9756, 0.9],
        'acc': [98.5, 91.67, 93.96]
    },
    {
        'throughput': [2.77, 3.56, 3.9],
        'power': [0.158, 0.2501, 0.271],
        'tiles': [0.387, 0.2882, 0.2662],
        'acc': [97.61, 90.62, 93.10]
    }
]


# models = ['LeNet-5', 'VGG-16', 'ResNet-20']
models = ['LeNet', 'VGG', 'ResNet']
names = [
         r'Baseline ($\alpha={},\gamma={}$)'.format(1, 0),
         r'Column-Combine ($\alpha={},\gamma={}$)'.format(8, 0),
         r'Column-Combine Pruning ($\alpha={},\gamma={}$)'.format(8, 0.5),
        ]
colors = ['coral', 'mediumseagreen', 'cornflowerblue']

width = 0.28 
w, h = 5*3.7, 6
fig, axs = plt.subplots(1, 4, figsize=(w, h))
ax = axs[0]
ax.yaxis.grid()
ax.set_axisbelow(True)
pos = np.arange(len(data))
for i in range(len(data)):
    ax.bar(pos + i*width, data[i]['throughput'], width, color=colors[i], label=names[i])

ax.set_ylabel('Throughput')
ax.set_xticks(pos+width)
ax.set_xticklabels(models)

ax = axs[1]
ax.yaxis.grid()
ax.set_axisbelow(True)
pos = np.arange(len(data))
for i in range(len(data)):
    ax.bar(pos + i*width, data[i]['tiles'], width, color=colors[i], label=names[i])

ax.set_ylabel('Number of Tiles')
ax.set_xticks(pos+width)
ax.set_xticklabels(models)

ax = axs[2]
ax.yaxis.grid()
ax.set_axisbelow(True)
pos = np.arange(len(data))
for i in range(len(data)):
    ax.bar(pos + i*width, data[i]['power'], width, color=colors[i], label=names[i])

ax.set_ylabel('Energy per Input Sample')
ax.set_xticks(pos+width)
ax.set_xticklabels(models)
plt.tight_layout()

ax = axs[3]
ax.yaxis.grid()
ax.set_axisbelow(True)
pos = np.arange(len(data))
for i in range(len(data)):
    ax.bar(pos + i*width, data[i]['acc'], width, color=colors[i], label=names[i])

ax.set_ylabel('Classification Accuracy (%)')
ax.set_ylim(85, 99)
ax.set_xticks(pos+width)
ax.set_xticklabels(models)

fig.subplots_adjust(top=0.85, left=0.08, right=0.99, bottom=0.08)  # create some space below the plots by increasing the bottom-value
axs.flatten()[-2].legend(loc='lower center', bbox_to_anchor=(-0.35, 1), ncol=3, handletextpad=0.15, columnspacing=1)

plt.savefig('figures/asic-bars.pdf', dpi=300)
plt.clf()
