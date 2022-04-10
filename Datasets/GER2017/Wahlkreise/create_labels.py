#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:34:46 2021

@author: fsvbach
"""

import pandas as pd

data = pd.read_csv('Datasets/GER2017/Wahlkreise/Wahlkreise.csv', delimiter=';',
                encoding='utf-8', header=5, usecols=[0,1,2])
data = data.dropna()
data.Nr = data.Nr.astype('int')

mask = data['gehört zu'] == 99

## Länder
länder = data.loc[mask, ['Nr', 'Gebiet']]
länder.set_index('Nr', inplace=True)
länder = länder.Gebiet.to_dict()

### Kreise
kreise = data.loc[~mask]
kreise.set_index('Nr', inplace=True)
kreise['gehört zu'] = kreise['gehört zu'].astype('int')

kreise['Bundesland'] = kreise['gehört zu'].map(länder)

namestr='Population density'
threshold = 1000

kreise['Label'] = "Western Germany"
kreise.loc[kreise['gehört zu'].isin([8,9]), 'Label'] = "Southern Germany"
kreise.loc[kreise['gehört zu']>11, 'Label'] = "Eastern Germany"

struktur = pd.read_csv('Datasets/GER2017/btw17_strukturdaten.csv', delimiter=';',
                encoding='ISO-8859-1', decimal=',', header=8, index_col=[0], usecols = [1,8])

struktur.columns=[namestr]
struktur = struktur.loc[struktur.index.get_level_values(0) < 500]
struktur.sort_values(by=namestr, inplace=True)
struktur['Label'] = struktur[namestr] > threshold

kreise.loc[struktur.loc[kreise.index, 'Label'], 'Label'] = 'Cities'
### Store as DataFrame
kreise.to_csv("Datasets/GER2017/Wahlkreise/labels.csv")
