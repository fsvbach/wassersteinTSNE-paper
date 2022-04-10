#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:14:52 2022

@author: fsvbach
"""

from Datasets.GER2017 import Bundestagswahl
import WassersteinTSNE as WT
import matplotlib.pyplot as plt
import itertools as it

GER     = Bundestagswahl()
labels  = GER.labeldict()
dataset = GER.data

Gaussians = WT.Dataset2Gaussians(dataset, normalize=False)
WSDM = WT.GaussianWassersteinDistance(Gaussians)
TSNE = WT.GaussianTSNE(WSDM, seed=9)

embedding = TSNE.fit(w=0.75, trafo=WT.RotationMatrix(90))
embedding.index = embedding.index.to_series().map(labels)

## all
selection = it.combinations(dataset.columns, r=2)
M,N = 3,5

## custom
selection=[('GRÜNE', 'SPD'),('SPD','AfD'),('AfD','CDU'), ('CDU', 'FDP'),('CDU', 'GRÜNE')]
M,N = 1,5

fig, axes = plt.subplots(M,N, figsize=(WT.ecml_textwidth, .25*WT.ecml_textwidth))

for ax, (feature1, feature2) in zip(axes.flatten(), selection):    
    corr = dataset.groupby(level=0).corr().fillna(0)
    sizes = corr.swaplevel().loc[feature1, feature2].values

    im=ax.scatter(embedding['x'], embedding['y'], s=.5,
               c=sizes, cmap='seismic', vmax=1, vmin=-1)
    if feature2 == "DIE LINKE":
        feature2="Linke"
    if feature1 == "DIE LINKE":
        feature1="Linke"
    if feature2 == "GRÜNE":
        feature2="Grüne"
    if feature1 == "GRÜNE":
        feature1="Grüne"
    ax.set_title(f'{feature1} $-$ {feature2}')
    ax.axis('off')
    print('Plotted Correlation')
    ax.set_aspect(0.49568)

fig.text(-.1,1, 'A', va='bottom', ha='left', weight='bold',  transform=axes[0].transAxes)

# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.set_ticks([-1,0,1])
# cbar.set_ticklabels(['anti', 'none', 'high'])
fig.tight_layout()
fig.savefig(f"Figures/Figure9A.pdf")


