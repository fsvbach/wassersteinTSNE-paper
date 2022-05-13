#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:17:24 2022

@author: bachmann
"""

#############  FIGURE 2  #################
from Experiments import Figure2

Figure2.Histogram()   # Figure 2A
Figure2.Uniform()     # Figure 2B
print("Created Figure 2")
##########################################


#############  FIGURE 3  #################
from Experiments import Figure3

#### Recompute Data for Figure 3 (takes ~18hrs)
#Figure3.recompute()

### or use cached results
Figure3.plot()
print("Created Figure 3")
##########################################


#############  FIGURE 4  #################
from Experiments import Figure4

Figure4.plot()
print("Created Figure 4")
##########################################


#############  FIGURE 5  #################
from Experiments import Figure5

Figure5.HGMM()                    #PANEL A
Figure5.Embed()                   #PANEL B-D
Figure5.Evaluate(recompute=False) #PANEL E (recomputation takes ~1min)
Figure5.save()
print("Created Figure 5")
##########################################


#############  FIGURE 6  #################
from Experiments import Figure6

Figure6.plot()
print("Created Figure 6")
##########################################


#############  FIGURE 7  #################
from Experiments import Figure7

Figure7.plot()
print("Created Figure 7")
##########################################


#############  FIGURE 8  #################
from Experiments import Figure8

Figure8.plot()
print("Created Figure 8")
##########################################


#############  FIGURE 9  #################
from Experiments import Figure9

Figure9.plot()                   #PANEL A
Figure9.plot(geographical=False) #PANEL B
print("Created Figure 9")
##########################################


#############  FIGURE 10  #################
from Experiments import Figure10

#### Recompute Distance Matrix (takes ~44hrs)
# Figure10.recomputeWasserstein()

#### Recompute Data for Panel C (takes ~30s)
# Figure10.recomputeAccuracies()

### or use cached results
Figure10.plot()
print("Created Figure 10")
##########################################