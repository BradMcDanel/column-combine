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


alpha = 8
gamma = 0.5
# sample_ratio = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0])
sample_ratio = np.array([0.025, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 1.0])
gstats = []
accs, effs, weights = {}, {}, {}
keys = ['baseline', 'pruned']
names = ['New Model', 'Pretrained Model']
for key in keys:
    accs[key] = []
    effs[key] = []
    weights[key] = []
    for r in sample_ratio:
        k = 'baseline' if r == 1.0 else key
        model_path = 'models/cifar10/cifar10-shift-{}-{}.pth'.format(k, r)
        if not os.path.exists(model_path):
            print(model_path)
            continue
        print(model_path)
        model = torch.load(model_path)
        accs[key].append(model.stats['test_acc'][-1])

colors = ['darkorchid', 'orangered']
markers = ['o', '^']
fig, ax1 = plt.subplots()
ax1.set_title('CIFAR-10 VGG-19 (1x1)')
for i, key in enumerate(keys):
    ax1.plot(100.*sample_ratio, accs[key], '-', marker=markers[i],
             color=colors[i], linewidth=3, ms=10, markeredgecolor='k', label=names[i])
ax1.set_xlabel('Fraction of Training Data (%)')
ax1.set_ylabel('Classification Accuracy (%)')
ax1.set_xscale('log')
ax1.set_xticks(100.*sample_ratio, [])
ax1.set_ylim((65, 100))
plt.minorticks_off()
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1.get_xaxis().set_minor_formatter(None)
# ax1.set_xticklabels(100.*sample_ratio,[])
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter('%g'))
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
plt.legend(loc='lower right')
# plt.grid()
plt.tight_layout()
plt.savefig('visualization/figures/limited.pdf', dpi=300)
plt.clf()
