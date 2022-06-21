#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:36:29 2022

@author: fsvbach
"""

import WassersteinTSNE as WT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Datasets.GER2017 import Bundestagswahl
from params import textwidth, colors


GER     = Bundestagswahl()
labeldict  = GER.labeldict('Label')
dataset = GER.data

########################### WASSERSTEIN #########################

def recomputeWasserstein():
    D = WT.WassersteinDistanceMatrix(dataset, timer=True)
    D.to_csv('Experiments/cache/ExactWassersteinDistances.csv')

######################### GAUSSIAN #############################

w_range = np.linspace(0,1,15)
    
def recomputeAccuracies():
    Gaussians = WT.Dataset2Gaussians(dataset)
    GWD = WT.GaussianWassersteinDistance(Gaussians)
    
    accuracies = []
    embeddings = {}
    
    for w in w_range:
        embedding = WT.ComputeTSNE(GWD.matrix(w=w), seed=9, perplexity = 30, trafo=WT.RotationMatrix(80)@WT.MirrorMatrix([0,1]))
    
        ari, _ = WT.LeidenClusters(GWD.matrix(w=w), labeldict, k=5, res=0.08, seed=9)
        knn, _ = WT.knnAccuracy(embedding, labeldict, k=5)
        embedding.index   = embedding.index.to_series(name='color').map(labeldict)
        embeddings[w] = embedding
        
        accuracies.append([knn, ari])
        
    accuracies = pd.DataFrame(accuracies, index=w_range, columns=['kNN']+[f'ARI'])
    accuracies.to_csv('Experiments/cache/GERresults.csv')
    embeddings[.5].to_csv('Experiments/cache/GERembedding.csv')
    
    
######################## EMBEDDING #####################
    
def plot():
    D = pd.read_csv('Experiments/cache/ExactWassersteinDistances.csv', index_col=0)
    D.columns = D.columns.astype(int)
    embeddingX = WT.ComputeTSNE(D, seed=13, trafo=WT.RotationMatrix(170))
    k = 5
    ARI, _ = WT.LeidenClusters(D, labeldict, seed=42, res=0.08, k=k)
    KNN, _ = WT.knnAccuracy(embeddingX, labeldict, k=k)
    embeddingX.index = embeddingX.index.to_series().map(labeldict)

    accuracies = pd.read_csv('Experiments/cache/GERresults.csv', index_col=0)
    embedding = pd.read_csv('Experiments/cache/GERembedding.csv', index_col=0)

    fig = plt.figure(figsize=(textwidth,.35*textwidth),dpi=300)
    
    ax1 = fig.add_axes([0.01,0.12,0.3,0.8])
    ax2 = fig.add_axes([0.32,0.12,0.3,0.8])
    ax3 = fig.add_axes([0.63,0.12,0.3,0.8])
    
    i = 0
    for label, data in embeddingX.groupby(level=0):
        X, Y = data['x'], data['y']
        ax1.scatter(X, Y, color=colors[i], label=label, s=4)
        i+=1
    ax1.set_xticks([]) # labels 
    ax1.set_yticks([])
    ax1.set_title(rf"exact")
    fig.text(0,1, 'A', va='bottom', ha='left', weight='bold', transform=ax1.transAxes)
    text = r'$\mathrm{kNN}$'+': %.2f\n'%KNN+r'$\mathrm{ARI}$: %.2f' %ARI
    ax1.text(0.99, 0.99, text, transform=ax1.transAxes, va='top', ha='right')
    
    i = 0
    for label, data in embedding.groupby(level=0):
        X, Y = data['x'], data['y']
        ax2.scatter(X, Y, color=colors[i], label=label, s=4)
        i+=1
    ax2.set_xticks([]) # labels 
    ax2.set_yticks([])
    ax2.set_title(rf"$\lambda$=0.5")
    fig.text(0,1, 'B', va='bottom', ha='left', weight='bold', transform=ax2.transAxes)
    idx = w_range==0.5
    
    knn = accuracies['kNN']
    ari = accuracies['ARI']
    ax3.axhline(y=100*ARI, color='black', linestyle='--')
    ax3.axhline(y=100*KNN, color='black', linestyle='--', label='exact')
    ax3.plot(w_range, 100*knn, marker='o', linewidth=1, markersize=2, label='kNN')
    ax3.plot(w_range, 100*accuracies[f'ARI'], marker='o', linewidth=1, markersize=2, label=f'ARI')
    
    ax3.set(ybound=(0,105),
            yticks=np.linspace(10,100,4),
            xticks=np.round(np.linspace(0,1,4),2))
    ax3.set_xlabel(r'$\lambda$',labelpad=-4)
    ax3.set_ylabel(r'$\%$',labelpad=-4)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.legend(handlelength=2.3)
    fig.text(0,1, 'C', va='bottom', ha='left', weight='bold', transform=ax3.transAxes)
    text = r'$\mathrm{kNN}$'+': %.2f\n'%knn[idx]+r'$\mathrm{ARI}$: %.2f' %ari[idx]
    ax2.text(0.99, 0.99, text, transform=ax2.transAxes, va='top', ha='right')
    
    
    handles, labels = ax1.get_legend_handles_labels()
    lgnd = fig.legend(handles, labels, loc=(0.01,0.03), 
                      fontsize=5, handletextpad=0.0, ncol=4, columnspacing=0.3)
    for handle in lgnd.legendHandles:
        handle._sizes =[25]
    
    fig.savefig(f"Figures/Figure10.pdf")
    return fig


if __name__ == '__main__':
    # recomputeWasserstein()
    # recomputeAccuracies()
    plot()