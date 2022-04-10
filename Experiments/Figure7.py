#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:20:04 2022

@author: fsvbach
"""


import igraph as ig
from sklearn.neighbors import kneighbors_graph
import leidenalg as la

import matplotlib.pyplot as plt

import WassersteinTSNE as WT
from Datasets.GER2017 import Bundestagswahl

shapes = ['1','.','*','s']
GER     = Bundestagswahl()
labeldict  = GER.labeldict('Label')
dataset = GER.data

# def embedding():
Gaussians = WT.Dataset2Gaussians(dataset)
WSDM = WT.GaussianWassersteinDistance(Gaussians)
TSNE = WT.GaussianTSNE(WSDM, seed=9, perplexity = 30)

accuracies = []
embeddings = []
w_range = [0,0.75,1]
trafos  =[WT.RotationMatrix(-110),WT.RotationMatrix(90),None]

for w, trafo in zip(w_range, trafos):
    
    k = 5
    res = 0.08
    A = kneighbors_graph(WSDM.matrix(w=w), k)
    sources, targets = A.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(A.shape[0])
    edges = list(zip(sources, targets))
    G.add_edges(edges)
    
    clustering = la.find_partition(G, la.RBConfigurationVertexPartition, 
                                      resolution_parameter = res, seed=9)
    

    embedding = TSNE.fit(w=w, trafo=trafo)
    embedding.index = embedding.index.to_series(name='color').map(labeldict)
    # embedding['sizes'] = 3
    embedding['labels']  = clustering.membership
    embeddings.append(embedding)
    accuracies.append([TSNE.knn_accuracy(w, labeldict, k=5),
                       TSNE.adjusted_rand_index(w, labeldict, k=5, res=0.08)])


width = WT.ecml_textwidth
height = .4*WT.ecml_textwidth
fig, axes = plt.subplots(1,3,figsize=(width,height))

mapping = {0:0,1:1,2:3,3:2}
for w, ax, embedding, acc, t in zip(w_range, axes, embeddings, accuracies, ['A','B','C']):
    i = 0
    for label, data in embedding.groupby(by='color'):
        # X, Y, c = data.values.T
        for c, XY in data.groupby(by='labels'):
            if w==0.75:
                c = mapping[c]
            ax.scatter(XY.x, XY.y, label=label, s=5,  color=WT.TUcolors[i], marker=shapes[c])
        i+=1
    ax.set_xticks([]) # labeldict 
    ax.set_yticks([])
    ax.set_title(f"$\lambda$={w}")
    fig.text(0,1, t, va='bottom', ha='left', weight='bold', transform=ax.transAxes)

    # ax.set_aspect('equal')
    
    text = r'$\alpha_{\mathrm{kNN}}$'+'=%.2f\n'%acc[0]+r'$\alpha_{\mathrm{ARI}}$=%.2f' %acc[1]
    # text = f'kNN={round(acc[0],2)}\nARI={round(acc[1],2)}'
    ax.text(0.99, 0.99, text, transform=ax.transAxes, va='top', ha='right')

handles, labels = axes[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
lgnd = fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(.5,-0.03), loc='lower center', ncol=4, handletextpad=0.1)
for handle in lgnd.legendHandles:
    handle._sizes =[20]

fig.tight_layout()
plt.subplots_adjust(wspace=0.1, 
                    hspace=0.0)
fig.savefig(f"Figures/Figure7.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
plt.close()
