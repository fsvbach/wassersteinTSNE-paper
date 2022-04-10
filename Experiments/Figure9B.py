#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:46:38 2022

@author: bachmann
"""

from Datasets.GER2017 import Bundestagswahl
import WassersteinTSNE as WT
import matplotlib.pyplot as plt
import itertools as it

GER     = Bundestagswahl()
labels  = GER.labeldict()
dataset = GER.data
map_wk  = GER.map()

## all
selection = it.combinations(dataset.columns, r=2)
M,N = 3,5

## custom
selection=[('GRÜNE', 'SPD'),('SPD','AfD'),('AfD','CDU'), ('CDU', 'FDP'),('CDU', 'GRÜNE')]
M,N = 1,5


fig, axes = plt.subplots(M,N, figsize=(WT.ecml_textwidth, .25*WT.ecml_textwidth))

for ax, (feature1, feature2) in zip(axes.flatten(), selection):    
    corr = dataset.groupby(level=0).corr().fillna(0)
    
    combined = map_wk.set_index("WKR_NR").join(corr.swaplevel().loc[feature1, feature2])
    
    combined.plot(column=feature2, cmap="seismic", linewidth=3, ax=ax, vmin=-1, vmax=1)
    
    # sm = plt.cm.ScalarMappable(cmap="seismic", 
    #                             norm=plt.Normalize(vmin=-1, 
    #                             vmax=1))
    #sm._A = []
    #cbar = fig.colorbar(sm, fraction=0.046, pad=0.04)

    print((combined.loc[:,feature2] >0).sum())
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
    
fig.text(-.2,1, 'B', va='bottom', ha='left', weight='bold',  transform=axes[0].transAxes)


# cbar = fig.colorbar(im, ax=axes.ravel().tolist())
# cbar.set_ticks([-1,0,1])
# cbar.set_ticklabels(['anti', 'none', 'high'])
fig.tight_layout()
fig.savefig(f"Figures/Figure9B.pdf")


