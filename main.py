#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:17:24 2022

@author: bachmann
"""

import WassersteinTSNE


#############  FIGURE 2  #################

print("Create Figure 2")
from Experiments import Figure2

Figure2.Histogram()   # Figure 2A
Figure2.Uniform()     # Figure 2B


#############  FIGURE 3  #################

print("Create Figure 3")
from Experiments import Figure3

#### Recompute Data for Figure 3
#Figure3.recompute()

### or use cached results
Figure3.plot()


#############  FIGURE 4  #################

print("Create Figure 4")
from Experiments import Figure4


#############  FIGURE 5  #################

print("Create Figure 5")
from Experiments import Figure5

Figure5.HGMM()                      #PANEL A
Figure5.Embed()                     #PANEL B-D
Figure5.Evaluate(recompute=False)   #PANEL E
Figure5.save()


#############  FIGURE 6  #################

print("Create Figure 6")
from Experiments import Figure6


#############  FIGURE 7  #################

print("Create Figure 7")
from Experiments import Figure7


#############  FIGURE 8  #################

print("Create Figure 8")
from Experiments import Figure8


#############  FIGURE 9  #################

print("Create Figure 9")
from Experiments import Figure9

Figure9.plot()                      #PANEL A
Figure9.plot(geographical=False)    #PANEL B


#############  FIGURE 10  #################

print("Create Figure 10")
from Experiments import Figure10

#### Recompute Data for Figure 10
#Figure10.recompute()

### or use cached results
Figure10.plot()