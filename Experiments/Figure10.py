#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:36:29 2022

@author: fsvbach
"""

import igraph as ig
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, cluster
import leidenalg as la

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import WassersteinTSNE as WT
from Datasets.GER2017 import Bundestagswahl

colors = ["C0","C1","C2", "C3"]
colors = WT.TUcolors

GER     = Bundestagswahl()
labeldict  = GER.labeldict('Label')
dataset = GER.data

##################################### WASSERSTEIN ############################################

A = pd.read_csv('Experiments/cache/ExactWassersteinDistances.csv', index_col=0)
A.columns = A.columns.astype(int)

tsne = WT.WassersteinTSNE(seed=13)
embeddingX = tsne.fit(A, trafo=WT.RotationMatrix(170))
embeddingX.index = embeddingX.index.to_series().map(labeldict)

k = 5
K = kneighbors_graph(A, k)
sources, targets = K.nonzero()
G = ig.Graph(directed=True)
G.add_vertices(K.shape[0])
edges = list(zip(sources, targets))
G.add_edges(edges)

clustering = la.find_partition(G, la.RBConfigurationVertexPartition, 
                                  resolution_parameter = 0.08, seed=42)
ARI = cluster.adjusted_rand_score(clustering.membership, embeddingX.index)


kNN    = KNeighborsClassifier(k+1)
kNN.fit(embeddingX.values, embeddingX.index)
test = kNN.predict(embeddingX.values)
ACC = accuracy_score(test, embeddingX.index)*(k+1)/k-1/k

#######################################################################GAUSS EMBEDDING ###########

w_range = np.linspace(0,1,15)
    
def recompute():
    Gaussians = WT.Dataset2Gaussians(dataset)
    WSDM = WT.GaussianWassersteinDistance(Gaussians)
    TSNE = WT.GaussianTSNE(WSDM, seed=9, perplexity = 30)
    
    accuracies = []
    embeddings = {}
    
    for w in w_range:
        embedding = TSNE.fit(w=w, trafo=WT.RotationMatrix(80)@WT.MirrorMatrix([0,1]))
        embedding.index = embedding.index.to_series(name='color').map(labeldict)
        embedding['sizes'] = 3
        embeddings[w] = embedding
        accuracies.append([TSNE.knn_accuracy(w, labeldict, k=5),
                            TSNE.adjusted_rand_index(w, labeldict, k=5, res=0.08)])

    accuracies = pd.DataFrame(accuracies, index=w_range, columns=['kNN']+[f'ARI{i}' for i in range(len(accuracies[0])-1)])
    accuracies.to_csv('Experiments/cache/GERresults.csv')
    embeddings[.5].to_csv('Experiments/cache/GERembedding.csv')
    
################################################ EMBEDDING ############################
    
def plot():
    accuracies = pd.read_csv('Experiments/cache/GERresults.csv', index_col=0)
    embedding = pd.read_csv('Experiments/cache/GERembedding.csv', index_col=0)
    
    width = WT.ecml_textwidth
    height = .35*WT.ecml_textwidth
    fig = plt.figure(figsize=(width,height),dpi=300)
    
    ax1 = fig.add_axes([0.01,0.12,0.3,0.8])
    ax2 = fig.add_axes([0.32,0.12,0.3,0.8])
    ax3 = fig.add_axes([0.63,0.12,0.3,0.8])
    
    i = 0
    for label, data in embeddingX.groupby(level=0):
        X, Y = data['x'], data['y']
        ax1.scatter(X, Y, color=WT.TUcolors[i], label=label, s=4)
        i+=1
    ax1.set_xticks([]) # labels 
    ax1.set_yticks([])
    ax1.set_title(rf"exact")
    fig.text(0,1, 'A', va='bottom', ha='left', weight='bold', transform=ax1.transAxes)
    text = r'$\alpha_{\mathrm{kNN}}$'+'=%.2f\n'%ACC+r'$\alpha_{\mathrm{ARI}}$=%.2f' %ARI
    ax1.text(0.99, 0.99, text, transform=ax1.transAxes, va='top', ha='right')
    
    i = 0
    for label, data in embedding.groupby(level=0):
        X, Y = data['x'], data['y']
        ax2.scatter(X, Y, color=WT.TUcolors[i], label=label, s=4)
        i+=1
    ax2.set_xticks([]) # labels 
    ax2.set_yticks([])
    ax2.set_title(rf"$\lambda$=0.5")
    fig.text(0,1, 'B', va='bottom', ha='left', weight='bold', transform=ax2.transAxes)
    idx = w_range==0.5
    
    knn = accuracies['kNN']
    ari = accuracies['ARI0']
    ax3.axhline(y=100*ARI, color='black', linestyle='--')
    ax3.axhline(y=100*ACC, color='black', linestyle='--', label='exact')
    ax3.plot(w_range, 100*knn, marker='o', linewidth=1, markersize=2, label='kNN')
    
    for i in range(0,1):
        ax3.plot(w_range, 100*accuracies[f'ARI{i}'], marker='o', linewidth=1, markersize=2, label=f'ARI{i}')
    
    ax3.set(ybound=(0,105),
            yticks=np.linspace(10,100,4),
            xticks=np.round(np.linspace(0,1,4),2))
    ax3.set_xlabel(r'$\lambda$',labelpad=-4)
    ax3.set_ylabel(r'$\%$',labelpad=-4)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.legend(handlelength=2.3)
    fig.text(0,1, 'C', va='bottom', ha='left', weight='bold', transform=ax3.transAxes)
    text = r'$\alpha_{\mathrm{kNN}}$'+'=%.2f\n'%knn[idx]+r'$\alpha_{\mathrm{ARI}}$=%.2f' %ari[idx]
    ax2.text(0.99, 0.99, text, transform=ax2.transAxes, va='top', ha='right')
    
    
    handles, labels = ax1.get_legend_handles_labels()
    lgnd = fig.legend(handles, labels, loc=(0.01,0.03), 
                      fontsize=5, handletextpad=0.0, ncol=4, columnspacing=0.3)
    for handle in lgnd.legendHandles:
        handle._sizes =[25]
    
    fig.savefig(f"Figures/Figure10.pdf")
    
    plt.show()