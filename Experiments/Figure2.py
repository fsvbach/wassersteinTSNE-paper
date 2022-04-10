#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:35:22 2022

@author: fsvbach
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import WassersteinTSNE as WT 
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

feature = 'v102'
dataset.index = dataset.index.to_series().map(labels)

name = 'B'

barheight = 0.1
barwidth  = 1 - barheight
ratio     = 1 / (2-barheight)

fig = plt.figure(dpi=300, figsize=(.5*WT.ecml_textwidth, .5*ratio*WT.ecml_textwidth))
ax2 = fig.add_axes([0,0,barheight*ratio,barwidth])
ax1 = fig.add_axes([barheight*ratio,1-barheight,barwidth*ratio,barheight])
ax3 = fig.add_axes([(barheight+barwidth)*ratio,0,barwidth*ratio,barwidth])
ax4 = fig.add_axes([barheight*ratio,0,barwidth*ratio,barwidth])

[ax.set_axis_off() for ax in [ax1,ax2,ax3,ax4]]

### HISTOGRAMM
if name == 'A':
    P1 = np.histogram(dataset.loc['de', feature], bins=10, density=False)
    P2 = np.histogram(dataset.loc['al', feature], bins=10, density=False)
    
    P1 = P1[0]/P1[0].sum()
    P2 = P2[0]/P2[0].sum()
    
    n = len(P1)
    minval, maxval = EVS.interval
    space, step  = np.linspace(minval,  maxval, n, retstep=True)
    
    D = WT.EuclideanDistance(space.reshape(-1,1), space.reshape(-1,1))
    
    A = WT.ConstraintMatrix(n,n)
    b = np.concatenate([P1,P2])
    c = D.reshape(-1)
        
    opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None], method='highs')
    
    emd = round(opt_res.fun,3)
    gamma = opt_res.x.reshape((n, n))
    
    ax1.bar(space, P1, width=step, color='C0', alpha=0.5)
    ax1.set(xlim=(minval-step/2, maxval+step/2))
    ax2.barh(space, P2, height=step, color='C1', alpha=0.5)
    ax2.set(ylim=(minval-step/2, maxval+step/2))
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax4.imshow(gamma.T, cmap='Greys', vmin=0)
    ax3.imshow(D, cmap='Greys', vmin=0)
    fig.text(0,1, 'A', va='top', ha='left', weight='bold')

### UNIFORM
if name == 'B':
    U = dataset.loc['de', feature][:45].values.reshape(-1,1)
    V = dataset.loc['al', feature][:45].values.reshape(-1,1)
    
    n,m = len(U), len(V)
    D = WT.EuclideanDistance(U, V)
    A = WT.ConstraintMatrix(n,m)
    b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
    c = D.reshape(-1)
    
    opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None], method='highs')
    emd = round( opt_res.fun,3)
    gamma = opt_res.x.reshape((n, m))
    
    ax1.bar(np.arange(n), 1/n, color='C0', alpha=0.5)
    ax1.set(xlim=(-0.5,n-0.5))
    ax2.barh(np.arange(m), 1/m, color='C1', alpha=0.5)
    ax2.set(ylim=(-0.5,m-0.5))
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax4.imshow(gamma.T, cmap='Greys', vmin=0)
    ax3.imshow(D.T, cmap='Greys', vmin=0)
    fig.text(0,1, 'B', va='top', ha='left', weight='bold')


fig.savefig(f"Figures/Figure2{name}.pdf")

