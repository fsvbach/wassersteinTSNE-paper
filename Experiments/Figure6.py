#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

import WassersteinTSNE as WT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Datasets.GER2017 import Bundestagswahl
from params import textwidth

GER = Bundestagswahl()
names   = GER.labeldict('Gebiet')
numbers = {v: k for k, v in names.items()}
dataset = GER.data

comparis = ['Mittelsachsen', 'Mittelems'], ['Berlin-Neukölln', 'Hamburg-Mitte']
features = ['AfD', 'Linke']
location = [0.19,0.31], [0.52, 0.03]

def plot():
    fig, axes = plt.subplots(1,2, figsize=(textwidth, .5*textwidth))
    
    for wahlkreise, ax, loc, t in zip(comparis, axes, location, ['A','B']):
            
        U = dataset.loc[numbers[wahlkreise[0]], features]
        V = dataset.loc[numbers[wahlkreise[1]], features]
        
        G = WT.GaussianDistribution()
        Gaussians = [G.estimate(U), G.estimate(V)]
        
        WSDM = WT.GaussianWassersteinDistance(pd.Series(Gaussians, index=wahlkreise))
        diff_gauss = WSDM.matrix().iloc[1,0]
        diff_means = WSDM.matrix(w=0).iloc[1,0]
        diff_covs = WSDM.matrix(w=1).iloc[1,0]    
                            
        opt_res = WT.PairwiseWassersteinDistance(U, V)
        diff_exact = np.sqrt(-opt_res.fun )
        
        U['color'] = "C0"
        V['color'] = "C1"
        A = pd.concat([U,V])
        X,Y,c =     A.sample(frac=1).values.T
        ax.scatter(X, Y, s=1, c=c, marker='.')
        
        for G, data, name, color in zip(Gaussians, [U,V], list(map(numbers.get, wahlkreise)), ['C0','C1']):
        
            correlation = round(data.corr().iloc[1,0],2)
        
            WT.plotGaussian(G, size=0, color='black', STDS=[2], ax=ax)
            
            x,y = G.mean
            ax.scatter(x,y,label=f'{names[name]}',c=color,s=20, edgecolor='black', zorder=4)
          
        ax.set_aspect('equal')    
        ax.set(#title='Correlation of poll stations within a voting district',
               xlabel='AfD',
               ylabel='Linke', 
               xlim=[0,.55],
               ylim=[0,.45],
               yticks=np.linspace(0,.40, 5))
        # ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)
        
        # place a text box in upper left in axes coords
        textstr = '\n'.join((
            r'$d_{{Covs}}=%.3f$' % (diff_covs, ),
            r'$d_{{Means}}=%.3f$' % (diff_means, ),
            r'$d_{{Gauss}}=%.3f$' % (diff_gauss, ),
            r'$d_{{Exact}}=%.3f$' % (diff_exact, )))
        
        ax.text(loc[0], loc[1], textstr, fontsize=6, va='bottom', ha='right')
        ax.legend(loc='best',  handletextpad=0.05)
        fig.text(0,1, t, va='bottom', ha='left', weight='bold', transform=ax.transAxes)
    
    fig.tight_layout()
    fig.savefig(f"Figures/Figure6.pdf", bbox_inches='tight', pad_inches=0.01)
    return fig

if __name__ == '__main__':
    plot()
