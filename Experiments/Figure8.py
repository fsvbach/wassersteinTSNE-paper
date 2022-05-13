#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:19:03 2022

@author: fsvbach
"""

import WassersteinTSNE as WT
import matplotlib.pyplot as plt

from Datasets.GER2017 import Bundestagswahl
from params import textwidth, colors, shapes

GER     = Bundestagswahl()
labeldict  = GER.labeldict('Label')
dataset = GER.data

accuracies = []
embeddings = []

trafos = [None, WT.RotationMatrix(90), WT.RotationMatrix(-80)@WT.MirrorMatrix([-1,1])]
diagonal = [False, True, False]
normalized = [False, False, True]

for diag, norm, trafo in zip(diagonal, normalized, trafos):
    
    Gaussians = WT.Dataset2Gaussians(dataset, diagonal = diag, normalize = norm)
    GWD = WT.GaussianWassersteinDistance(Gaussians)

    ari, clusters = WT.LeidenClusters(GWD.matrix(w=1), labeldict, k=5, res=0.08, seed=9)
    embedding = WT.ComputeTSNE(GWD.matrix(w=1), trafo=trafo, seed=9)
    knn, _ = WT.knnAccuracy(embedding, labeldict, k=5)
    
    embedding['labels']   = embedding.index.to_series(name='color').map(labeldict)
    embedding['clusters'] = clusters
    embeddings.append(embedding)
    
    accuracies.append([knn, ari])

def plot():
    fig, axes = plt.subplots(1,3,figsize=(textwidth,.4*textwidth))
      
    mapping = {0:3,1:0,3:1,2:2}
    names = ['full covariance ($\lambda$=1)','diagonalized','normalized']
    for ax, embedding, acc, t, name in zip(axes, embeddings, accuracies, ['A','B','C'], names):
        i = 0
        for label, data in embedding.groupby(by='labels'):
            # X, Y, c = data.values.T
            for c, XY in data.groupby(by='clusters'):
                ax.scatter(XY.x, XY.y, label=label, s=5, color='C'+str(c), marker=shapes[mapping[i]])
            i+=1
        ax.set_xticks([]) # labeldict 
        ax.set_yticks([])
        ax.set_title(name)
        fig.text(0,1, t, va='bottom', ha='left', weight='bold', transform=ax.transAxes)
    
        # ax.set_aspect('equal')
        
        text = r'$\alpha_{\mathrm{kNN}}$'+'=%.2f\n'%acc[0]+r'$\alpha_{\mathrm{ARI}}$=%.2f' %acc[1]
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
    fig.savefig(f"Figures/Figure8.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    return fig

if __name__ == '__main__':
    plot()
