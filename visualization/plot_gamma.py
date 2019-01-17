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

colors = ['r','g','b']
alpha = 8
beta = 0.7
gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
gstats = []
for gamma in gammas:
    model_path = 'models/cifar10/cifar10-shift-{}.pth'.format(gamma)
    model = torch.load(model_path).cpu()
    macs = np.sum([x*y for x, y in model.packed_layer_size])
    curr_weights, _ = util.num_nonzeros(model)
    gstats.append(model.stats)

accs, effs, weights = [], [], []
for i, gamma in enumerate(gammas):
    accs.append(gstats[i]['test_acc'][-1])
    effs.append(gstats[i]['efficiency'][-1])
    weights.append(gstats[i]['weight'][-1])

fig, ax1 = plt.subplots()
ax1.set_title('CIFAR-10 VGG-19 (1x1)')
ln1 = ax1.plot(gammas, accs, '-o', color=colors[2], linewidth=3, ms=10, markeredgecolor='k', label='Classification Accuracy (%)')
ax1.set_xlabel(r'$\gamma$')
ax1.set_ylim((92, 94))
ax1.set_ylabel('Classification Accuracy (%)', color='b')
ax1.set_xticks(gammas)

ax2 = ax1.twinx()
ln2 = ax2.plot(gammas, effs, '-^', color=colors[1], linewidth=3, ms=10, markeredgecolor='k', label='Packing Efficiency (%)')
ax2.set_xlabel(r'$\gamma$')
ax2.set_ylabel('Utilization Efficiency (%)', color='g')

# lns = ln1 + ln2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='lower right')

fig.tight_layout()
plt.savefig('visualization/figures/gamma-acc-eff.pdf', dpi=300)
plt.clf()


fig, ax1 = plt.subplots()
ax1.set_title(r'ResNet-20: $\alpha={},\beta={:0.1f}$'.format(alpha, 1-beta))
ax1.plot(gammas, accs, '-ob', linewidth=2)
ax1.set_xlabel(r'$\gamma$')
ax1.set_ylabel('Classification Accuracy (%)', color='b')
ax1.set_xticks(gammas)

ax2 = ax1.twinx()
ax2.plot(gammas, weights, '-or', linewidth=2)
ax2.set_xlabel(r'$\gamma$')
ax2.set_ylabel('Nonzero Weights', color='r')

fig.tight_layout()
plt.savefig('visualization/figures/gammma-acc-weights.pdf', dpi=300)
plt.clf()

fig, ax1 = plt.subplots()
ax1.set_title(r'ResNet-20: $\alpha={},\beta={:0.1f}$'.format(alpha, 1-beta))
ax1.plot(gammas, weights, '-or', linewidth=2)
ax1.set_xlabel(r'$\gamma$')
ax1.set_ylabel('Classification Accuracy (%)', color='b')
ax1.set_ylabel('Nonzero Weights', color='r')
ax1.set_xticks(gammas)


ax2 = ax1.twinx()
ax2.plot(gammas, effs, '-og', linewidth=2)
ax2.set_xlabel(r'$\gamma$')
ax2.set_ylabel('Packing Efficiency (%)', color='g')




fig.tight_layout()
plt.savefig('visualization/figures/gammma-weight-eff.pdf', dpi=300)
plt.clf()