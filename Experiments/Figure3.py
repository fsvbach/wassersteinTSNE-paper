#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:58:18 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import WassersteinTSNE as WT
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

timer = WT.Timer("Gaussian Wasserstein")

G1 = WT.GaussianDistribution(mean= np.array([10,15]),
                              cov = WT.CovarianceMatrix(WT.RotationMatrix(-45), s=[25,3]))
G2 = WT.GaussianDistribution(mean= np.array([20,15]),
                             cov = WT.CovarianceMatrix(WT.RotationMatrix(-135), s=[50,5]))

WSDM = WT.GaussianWassersteinDistance(pd.Series([G1,G2], index=['Blue', 'Orange']))

dist_e = WSDM.matrix(w=0)[1,0]
dist_c = WSDM.matrix(w=1)[1,0]
dist_w = WSDM.matrix()[1,0]

samplesizes = np.arange(1,21)*50
testsize = 50

timer.add(f'Generated two Gaussians')

# results = np.zeros(shape=(len(samplesizes),testsize))
# cctimes = np.zeros(shape=(len(samplesizes),testsize))

# for i, n in enumerate(samplesizes):
#     for j in range(testsize):
#         time = timer.time()
#         U = G1.samples(size=n)
#         V = G2.samples(size=n)
#         opt_res = linprogSolver(U, V)
#         results[i,j] = np.sqrt(-opt_res.fun )
#         cctimes[i,j] = timer.time() - time
#         print(results[i,j])
#     timer.add(f'Computed {testsize} distances with {n} samples')

# np.save('Experiments/cache/GaussianDistances', results)
# np.save('Experiments/cache/GaussianTimes', cctimes)
results = np.load('Experiments/cache/GaussianDistances.npy')
cctimes = np.load('Experiments/cache/GaussianTimes.npy')

means = results[:-1].mean(axis=1)
stds  = results[:-1].std(axis=1)
times = cctimes[:-1].mean(axis=1)
tstds  = cctimes[:-1].std(axis=1)

samplesizes=samplesizes[:-1]
# timer.add(f'Saved distance matrix and times')


fig = plt.figure(figsize=(WT.ecml_textwidth, .35*WT.ecml_textwidth), dpi=300)
ax = plt.subplot(1, 2, 1)
res1 = plt.subplot(1, 2, 2)

# fig, (ax, res1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
res2 = res1.twinx()

C0 = 'C0' # WT.TUgold
C1 = 'C1' # WT.imprsdb
WT.plotGaussian(G1, ax=ax, STDS=[2], color=C0, lw=1, r=0.5, size=testsize)
WT.plotGaussian(G2, ax=ax, STDS=[2], color=C1, lw=1, r=0.5, size=testsize)

ax.set_aspect('equal')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

ax.scatter(G1.mean[0], G1.mean[1], color=C0, s=25)
ax.scatter(G2.mean[0], G2.mean[1], color=C1, s=25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.text(-.13,1, 'A', va='bottom', ha='left', weight='bold', transform=ax.transAxes)

C2 = 'C2' # WT.TUdark
#res.plot(samplesizes, [dist_c for i in samplesizes],  color=c, linestyle='dashed')
#res.annotate('Euclidean Distance', (400, dist_e+0.1), fontsize=7)
#res.annotate('Covariance Distance', (200, dist_c+0.1), fontsize=9)

#res1.set_title('estimated 2-Wasserstein distance')
#res2.set_title('computation time [s]')
l1 = res1.plot(samplesizes, means, linewidth=2, color=C2)
res1.fill_between(samplesizes, means+stds, means-stds, color=C2, alpha=0.3, edgecolor='none')
res1.set_ylabel('2-Wasserstein distance', color=C2)
res1.set_xlabel('sample size', labelpad=1)
res1.set(ylim=(10.5,13),
         xlim=(50,950),
         yticks=(11,11.5,12,12.5))
res1.tick_params(axis='y', colors=C2)  
fig.text(-.28,1, 'B', va='bottom', ha='left', weight='bold', transform=res1.transAxes)

# res1.yaxis.set_label_position("right")
# res1.yaxis.tick_right()
# res2.yaxis.set_label_position("right")
# res2.yaxis.tick_right()

c='black' # WT.Tured
# res1.plot(samplesizes, [dist_e for i in samplesizes],  color=c, linestyle='dashed')
l2 = res1.plot(samplesizes, 
          [dist_w for i in samplesizes],  
          color=c, 
          lw=1,
          linestyle='dashed')

C3 = 'C4' # WT.mpgreen
l3 = res2.plot(samplesizes, times, linewidth=1, color=C3)
res2.fill_between(samplesizes, times+tstds, times-tstds, color=C3, alpha=0.3, edgecolor='none')
res2.set_ylabel('computation time [s]', color=C3)
res2.spines['right'].set_color(C3)
res2.tick_params(axis='y', colors=C3)  
res2.spines['left'].set_color(C2)
#res2.set_ylim(0,420)
res2.set_yscale('log')
res2.set_xscale('log')

lns = l1+l2+l3
labs = [l.get_label() for l in lns]
#res.legend(lns, labs, loc=(0.05,0.3))
res1.spines['top'].set_visible(False)
res2.spines['top'].set_visible(False)

for n, Z in zip([3,4],[100**3,250**4]):
    res2.plot(samplesizes, samplesizes**n/Z, 
              label=n, 
              c='gray',
              lw=.2,
              linestyle='-')
res2.annotate('x³', (80, 0.005), c='gray', fontsize=7)
res2.annotate('x²', (700, 750), c='gray', fontsize=7)
res2.set_xticks([50, 100, 400, 950])
res2.get_xaxis().set_major_formatter(ScalarFormatter())
# res2.get_xaxis().get_major_formatter().labelOnlyBase = False
fig.tight_layout()
fig.savefig('Figures/Figure3.pdf')#, bbox_inches='tight', pad_inches=0.0)
#fig.savefig('Reports/Figures/Wasserstein/ComputationDemo.pdf')
plt.show()
plt.close()

# timer.add('Plotted Figure')
# timer.finish('Plots/.logfile.txt')
