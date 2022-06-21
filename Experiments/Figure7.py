#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:20:04 2022

@author: fsvbach
"""

import WassersteinTSNE as WT
import matplotlib.pyplot as plt

from Datasets.GER2017 import Bundestagswahl
from params import textwidth, colors, shapes

GER     = Bundestagswahl()
labeldict  = GER.labeldict('Label')
dataset = GER.data
w_range = [0,0.75,1]

Gaussians = WT.Dataset2Gaussians(dataset)
GWD = WT.GaussianWassersteinDistance(Gaussians)

accuracies = []
embeddings = []
trafos  =[WT.RotationMatrix(-110),WT.RotationMatrix(90),None]

for w, trafo in zip(w_range, trafos):
    
    ari, clusters = WT.LeidenClusters(GWD.matrix(w=w), labeldict, k=5, res=0.08, seed=9)

    embedding = WT.ComputeTSNE(GWD.matrix(w=w), trafo=trafo, seed=9)
    knn, _ = WT.knnAccuracy(embedding, labeldict, k=5)
    
    embedding['labels']   = embedding.index.to_series().map(labeldict)
    embedding['clusters'] = clusters
    embeddings.append(embedding)
    
    accuracies.append([knn, ari])

def plot():
    fig, axes = plt.subplots(1,3,figsize=(textwidth,.4*textwidth))
    
    colormapping = {0:0,1:1,2:3,3:2}
    labelmapping = {0:3,1:0,2:2,3:1}
    for w, ax, embedding, acc, t in zip(w_range, axes, embeddings, accuracies, ['A','B','C']):
        i = 0
        for label, data in embedding.groupby(by='labels'):
            # X, Y, c = data.values.T
            for c, XY in data.groupby(by='clusters'):
                if w==0.75:
                    c = colormapping[c]
                ax.scatter(XY.x, XY.y, label=label, s=5,  color='C'+str(c), marker=shapes[labelmapping[i]])
            i+=1
        ax.set_xticks([]) # labeldict 
        ax.set_yticks([])
        ax.set_title(f"$\lambda$={w}")
        fig.text(0,1, t, va='bottom', ha='left', weight='bold', transform=ax.transAxes)
    
        # ax.set_aspect('equal')
        
        text = r'$\mathrm{kNN}$'+': %.2f\n'%acc[0]+r'$\mathrm{ARI}$: %.2f' %acc[1]
        # text = f'kNN={round(acc[0],2)}\nARI={round(acc[1],2)}'
        ax.text(0.99, 0.99, text, transform=ax.transAxes, va='top', ha='right')
    
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgnd = fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(.5,-0.03), loc='lower center', ncol=4, handletextpad=0.1)
    for handle in lgnd.legendHandles:
        handle._sizes =[20]
        handle.set_color('black')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, 
                        hspace=0.0)
    fig.savefig(f"Figures/Figure7.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02)
    return fig

if __name__ == '__main__':
    plot()


