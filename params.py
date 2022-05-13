#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:58:48 2022

@author: bachmann
"""

import matplotlib.pyplot as plt

inches_per_pt = 1 / 72.27
textwidth = 347.12354 * inches_per_pt

gray  = (155 / 255, 155 / 255, 155 / 255) 
green = (0 / 255, 108 / 255, 102 / 255)
blue = (35 / 255, 127 / 255, 154 / 255) 
red   = (141 / 255, 45 / 255, 57 / 255) 
yellow  = (174 / 255, 159 / 255, 109 / 255)
black  = (55 / 255, 65 / 255, 74 / 255) 
colors = {0: red, 1:blue, 2:green, 3:yellow, 4:gray, 5:black}
shapes  = ['1','.','*','s']

plt.rcParams.update({
    "figure.figsize": (textwidth, .35*textwidth),
    "figure.dpi": 300,
    # "text.usetex": True,                
    # "font.family": "sans-serif",
    # 'mathtext.fontset': 'cm',
    # "text.latex.preamble" : r"\usepackage{cmbright}",
    # 'font.sans-serif': ['Computer Modern'],
    # # "text.usetex": False, # <- don't use LaTeX to typeset. It's much slower, and you can't change the font atm.
    # "pgf.texsystem": "pdflatex",
    "axes.labelsize": 7, 
    "font.size": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    'lines.linewidth': 1,
    "axes.facecolor":'none',
    'axes.titlesize': 7,
    'axes.titlepad' : 1,    
    'axes.linewidth': 0.5})