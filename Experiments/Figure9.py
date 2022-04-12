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
map_wk  = GER.mapdict()

Gaussians = WT.Dataset2Gaussians(dataset, normalize=False)
WSDM = WT.GaussianWassersteinDistance(Gaussians)
TSNE = WT.GaussianTSNE(WSDM, seed=9)

embedding = TSNE.fit(w=0.75, trafo=WT.RotationMatrix(90))
embedding.index = embedding.index.to_series().map(labels)


def plot(geographical=True, selection=True):
    
    if selection:
        selection=[('Grüne', 'SPD'),('SPD','AfD'),('AfD','CDU'), ('CDU', 'FDP'),('CDU', 'Grüne')]
        M,N = 1,5
    else: 
        selection = it.combinations(dataset.columns, r=2)
        M,N = 3,5
    
    fig, axes = plt.subplots(M,N, figsize=(WT.ecml_textwidth, .25*WT.ecml_textwidth))
    
    for ax, (feature1, feature2) in zip(axes.flatten(), selection):    
        corr = dataset.groupby(level=0).corr().fillna(0)
        
        if geographical:
            panel='B'
            combined = map_wk.set_index("WKR_NR").join(corr.swaplevel().loc[feature1, feature2])
            combined.plot(column=feature2, cmap="seismic", linewidth=3, ax=ax, vmin=-1, vmax=1)
        else:
            panel='A'
            sizes = corr.swaplevel().loc[feature1, feature2].values
            im=ax.scatter(embedding['x'], embedding['y'], s=.5,
                       c=sizes, cmap='seismic', vmax=1, vmin=-1)
            ax.set_aspect(0.49568)
            
        ax.set_title(f'{feature1} $-$ {feature2}')
        ax.axis('off')
        print('Plotted Correlation')
        
    fig.text(-.2,1, panel, va='bottom', ha='left', weight='bold',  transform=axes[0].transAxes)
    
    fig.tight_layout()
    fig.savefig(f"Figures/Figure9{panel}.pdf")
    

if __name__ == '__main__':
    plot()
    plot(geographical=False)
