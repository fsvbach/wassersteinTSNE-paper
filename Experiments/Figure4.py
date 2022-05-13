#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:41:27 2021

@author: fsvbach
"""

import WassersteinTSNE as WT
import matplotlib.pyplot as plt
import numpy as np

from params import textwidth

def plot():
    figure, ax = plt.subplots(dpi=300, figsize=(textwidth, 0.45*textwidth))
    mixture = WT.HierarchicalGaussianMixture(seed=3, classes=4, datapoints=100, samples=15)
    trafo = np.array([[2,0],[0,2]])
    for i in range(4):
        mixture.ClassGaussians[i].cov.P = mixture.ClassGaussians[i].cov.P @ trafo
    mixture.ClassGaussians[2].mean += np.array([5,10])
    mixture.ClassGaussians[0].mean += np.array([-15,20])
    mixture.ClassGaussians[1].mean += np.array([5,-5])
    mixture.ClassGaussians[3].mean += np.array([10,-5])
    trafo = np.array([[2,0],[0,1]])
    for i in range(4):
        mixture.ClassGaussians[i].mean = mixture.ClassGaussians[i].mean @ trafo
        mixture.ClassWisharts[i].nu = 4
    mixture.generate_data()
    WT.plotMixture(mixture, std=4, ax=ax)
    limits = np.vstack((mixture.data.min(),mixture.data.max()))
    L = limits.mean(axis=0)
    (xmin, ymin), (xmax, ymax) = (limits-L)*(1.1,1.1)+L
    figure.axes[0].set(title='', xticks=[],yticks=[],  xlim=(xmin,xmax), ylim=(ymin, ymax), aspect='equal')
    figure.tight_layout()
    figure.savefig("Figures/Figure4.pdf",  bbox_inches='tight', pad_inches=0.02)
    plt.show()

if __name__ == '__main__':
    plot()