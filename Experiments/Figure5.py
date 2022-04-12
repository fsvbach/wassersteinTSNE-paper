#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 14:20:26 2022

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import WassersteinTSNE as WT


ylim=[-35,35]
xlim=[-20,50]
ls, htp = 0.8, 0.8
mixture = WT.HierarchicalGaussianMixture(seed=13,
                                        datapoints=100, 
                                        samples=30, 
                                        features=2, 
                                        classes=4,
                                        random=False)

C = WT.CovarianceMatrix(WT.RotationMatrix(20), s=[10,0.5])
D = WT.CovarianceMatrix(WT.RotationMatrix(110), s=[10,0.5])

mixture.set_params(means   = np.array([[30,0],[30,0],[0,0],[0,0]]),
                   Gammas = [WT.CovarianceMatrix(s=[5,5])]*4,
                   nus     = np.ones(4)*4,
                   Lambdas  = [C,D,C,D])

mixture.generate_data()
Gaussians = WT.Dataset2Gaussians(mixture.data)
GWD = WT.GaussianWassersteinDistance(Gaussians)
TSNE = WT.GaussianTSNE(GWD, seed=9)
print('Generated Mixture')


ratio= .5
xpad = 3/100
ypad = xpad/ratio
x    = (.5-1.5*xpad)/2
y    = (1-3*ypad)/2
width = WT.ecml_textwidth
height= ratio*width
fig = plt.figure(dpi=300, figsize=(width, height))
fig.text(xpad/2,1-ypad,'A', va='bottom', ha='left', fontweight='bold')


def HGMM():
    hgm = fig.add_axes([0,ypad,.5-1*xpad,1-2*ypad])

    # flattening the dataset
    datlabel = 'Unit means'
    # covlabel = rf'{std}-$\sigma$ class covariance'
    covlabel = 'Classes'
    dataset  = mixture.data.groupby(level=0).mean().sample(frac=1, random_state=43)
    dataset.index = dataset.index.to_series().map(mixture.labeldict())
    
    
    # plotting colourful datapoints
    xmeans, ymeans = dataset.values.T
    hgm.scatter(xmeans, ymeans, s=1, c=dataset.index, label=datlabel, zorder=3)
    datlabel=None
    
    #plotting samples
    xsample, ysample = mixture.data.values.T
    hgm.scatter(xsample, ysample, s=.5, linewidths=0, c='grey', label='Individual samples', zorder=2)
    
    # for Gaussian in mixture.ClassGaussians:
    #     # plotting black class covariances
    #     mean, width, height, angle = Gaussian.shape(std=std)
    #     ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
    #                   edgecolor='black', facecolor='none', 
    #                   linewidth=2, linestyle='--', 
    #                   label=covlabel, zorder=1)
    #     ax.add_patch(ell)
    #     covlabel=None
    
    # storing 1st legend 
    leg1 = hgm.legend(handler_map={WT.Ellipse: WT.HandlerEllipse()}, handletextpad=1,
                     loc='upper right', title="Hierarchical Structure",
                     facecolor='white', scatterpoints=4, framealpha=1)    
    
    leg1.legendHandles[1]._sizes = [2.5]
    
    handles = []
    labels  = []
    for i, Wishart in enumerate(mixture.ClassWisharts):
        # adding data covariances to 2nd legend 
        width, height, angle = Wishart.shape(std=2)
        ell = WT.Ellipse(xy=(0,0), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=.5, 
                      label='class '+str(i+1))
        handles.append(ell)
        labels.append('Class ' +str(i))
        # ax.add_artist(ell)
        
    # adding legends
    leg2 = hgm.legend(handles, labels, handler_map={WT.Ellipse: WT.HandlerEllipseRotation()},
                      title="Wishart Scales", loc=("lower left"), ncol=int(np.ceil(mixture.K/2)), facecolor='white',
                      labelspacing=ls, columnspacing=1, handleheight=1, handlelength=1, handletextpad=htp, borderpad=0.6)
    leg2.get_frame().set_linewidth(.5)
    leg2._legend_box.align = "left"
    leg1.get_frame().set_linewidth(.5)
    hgm.add_artist(leg1)

    hgm.set(title='',
            xlim=xlim,ylim=ylim,
            xticks=[],yticks=[])
    hgm.set_aspect('equal')

    print('Plotted Mixture')

### Embed three ScatterPlots
def Embed():
    
    for w, a, b, t in zip([0,0.5,1], [0.5,0.5+x+xpad/2,0.5], [2*ypad+y, 2*ypad+y,ypad], ['B','C','D']):
        ax = fig.add_axes([a-xpad/2,b,x,y]) 
        embedding = TSNE.fit(w=w).sample(frac=1, random_state=43)
        embedding.index = embedding.index.to_series().map(mixture.labeldict())
        ax.scatter(embedding['x'], embedding['y'], s=2, linewidth=0, c=embedding.index)
        ax.set(xticks=[],yticks=[])
        fig.text(a-xpad/2+x/2,b+y,rf"$\lambda$={w}", va='bottom', ha='center')
        fig.text(a-xpad/2,b+y,t, va='bottom', ha='left', fontweight='bold')

        # ax.axis('off')
        
    print('Plotted embeddings')
    
    
### Calculate Accuracies
def Evaluate(recompute=False):
    acc = fig.add_axes([0.5+x,ypad,x,y]) 

    if recompute:
        accs = []
        aris = []
        w_range= np.linspace(0,1,15)
        for w in w_range:
            accuracy = TSNE.knn_accuracy(w, mixture.labeldict(), k=5)
            accs.append(accuracy*100) 
            accuracy = TSNE.adjusted_rand_index(w, mixture.labeldict(),k=5,res=0.08)
            aris.append(accuracy*100) 
        results = pd.DataFrame(np.array([accs,aris]).T,index=w_range, columns=['kNN','ARI'])
        results.to_csv(f'Experiments/cache/HGMMresults.csv')
    else:
        results = pd.read_csv(f"Experiments/cache/HGMMresults.csv", index_col=0)
        
    acc.plot(results.index, results.kNN, marker='o', linewidth=1, markersize=2,label='kNN')
    acc.plot(results.index, results.ARI, marker='o', linewidth=1, markersize=2, label='ARI')
    acc.set(ybound=(0,105),
            yticks=np.linspace(10,100,4),
            xticks=np.round(np.linspace(0,1,4),2))
    acc.set_xlabel(r'$\lambda$',labelpad=-4)
    acc.set_ylabel(r'$\%$',labelpad=-4)
    # acc.set_title(f"evaluation (k=5)")
    # acc.tick_params(axis="y",direction="in", pad=-22)
    # acc.tick_params(axis="x",direction="in", pad=-15)
    # # acc.tick_params(axis="y",direction="in", pad=-22)
    acc.yaxis.tick_right()
    acc.yaxis.set_label_position("right")
    # acc.legend(loc=(0.45,0.05)) 
    acc.legend(loc='best')
    # acc.axis('off')
    # fig.text(0.5+1.5*x,ypad+y,rf"$\lambda$={w}", va='bottom', ha='center')
    fig.text(0.5+x,ypad+y, 'E', va='bottom', ha='left', weight='bold')
    acc.tick_params(axis='both', which='major', pad=1)


def save():
    fig.savefig(f"Figures/Figure5.pdf")
    return fig


if __name__ == '__main__':
    HGMM()
    Embed()
    Evaluate()
    save()
