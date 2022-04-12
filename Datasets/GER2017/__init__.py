#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:02:33 2021

@author: fsvbach
"""

import pandas as pd
import geopandas as gpd

class Bundestagswahl:
    data = pd.read_csv('Datasets/GER2017/btw17_wbz_zweitstimmen.csv', delimiter=';',
                    encoding='ISO-8859-1', header=4, index_col=[0], usecols = [0,8]+list(range(16,24)))
     
    ### Merging CDU and CSU
    data['CDU'] += data['CSU']
    data.drop('CSU', axis=1, inplace=True)
    
    ### Entferne leere Wahlkreise
    data = data.loc[data.Gültige > 20]
    
    ### Briefwähler
    # letter = data.loc[data.Bezirksart != 0]
    # letter.drop('Bezirksart', axis=1, inplace=True)
    # data = data.loc[data.Bezirksart == 0]
    data.drop('Bezirksart', axis=1, inplace=True)

    ### In Prozent umrechnen
    data = data.divide(data.Gültige, axis='rows')
    data.drop('Gültige', axis=1, inplace=True)
    # letter = letter.divide(letter.Gültige, axis='rows')
    # letter.drop('Gültige', axis=1, inplace=True)
    
    ### rename parties
    names = dict(zip(data.columns,data.columns))
    names['GRÜNE'] = 'Grüne'
    names['DIE LINKE'] = 'Linke'
    data.columns = data.columns.map(names)
    
    def labeldict(self,column='Bundesland'):
        labels = pd.read_csv('Datasets/GER2017/Wahlkreise/labels.csv',
                          index_col=0)
        return labels[column].to_dict()

    def mapdict(self):
        return gpd.read_file("Datasets/GER2017/Wahlkreise/Geometrie_Wahlkreise_20DBT_geo.shp")
