from __future__ import print_function

import argparse
import os
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
import measurement
import matplotlib
import matplotlib.pyplot as plt
from config import init_mpl_settings
init_mpl_settings()

colors = ['r','g','b']
alphas = [1, 2, 4, 8, 16]
beta = 0.7
gamma = 0.5
gstats = []
for alpha in alphas:
    model_path = 'models/c10-v2/{}-{}-{}.pth'.format(alpha, beta, gamma)
    model = torch.load(model_path).cpu()
    macs = np.sum([x*y for x, y in model.packed_layer_size])
    curr_weights, _ = util.num_nonzeros(model)
    gstats.append(model.stats)

accs, effs, weights = [], [], []
for i, alpha in enumerate(alphas):
    accs.append(gstats[i]['acc'][-1])
    effs.append(gstats[i]['efficiency'][-1])
    weights.append(gstats[i]['weight'][-1])

fig, ax1 = plt.subplots()
ax1.set_title(r'ResNet-20: $\beta={},\gamma={}$'.format(20, gamma))
ln1 = ax1.plot(alphas, accs, '-o', color=colors[2], linewidth=3, ms=10, markeredgecolor='k', label='Classification Accuracy (%)')
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylim((90, 95))
ax1.set_ylabel('Classification Accuracy (%)', color='b')
ax1.set_xticks(alphas)

ax2 = ax1.twinx()
ln2 = ax2.plot(alphas, effs, '-^', color=colors[1], linewidth=3, ms=10, markeredgecolor='k', label='Packing Efficiency (%)')
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('Utilization Efficiency (%)', color='g')

fig.tight_layout()
plt.savefig('figures/alpha-acc-eff.pdf', dpi=300)
plt.clf()