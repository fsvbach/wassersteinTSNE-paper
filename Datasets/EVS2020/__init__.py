#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class EuropeanValueStudy:
    interval = (1,10)
    
    UP   = (1,10)
    DOWN = (10,1)
    up   = (1, 5)
    down = (5, 1)
    
    legend =  {'v38': ('I have complete control over my life', UP),
                 'v39': ('I am satisfied with my life', UP), #free choice
                 'v63': (' God is important in my life', UP),# religion
                 'v102': ("Do you consider yourself 'left' or right'?", UP), # political landscape
                 'v103': ('Everyone is responsible for him/herself', DOWN),
                 'v104': ('The unemployed should take any job', DOWN),
                 'v105': ('Competition is good', DOWN),
                 'v106': ('Incomes should be made equal', DOWN),
                 'v107': ('Private ownership should be increased', DOWN),  # social welfare
                 'v143': ('My country is governed democratically', UP),
                 'v144': ('I am satisfied with the government', UP),        # satisfaction with status quo
                 'v185': ('Immigrants take jobs away', DOWN), 
                 'v186': ('Immigrants make crime problems', DOWN), 
                 'v187': ('Immigration is a strain on welfare system', DOWN),
                 'v188': ('Immigrants should maintain their traditions', DOWN), # immigration
                 'v199': ('I would give money for the environment', down),
                 'v200': ('Someone like me can do much for environment', up),
                 'v201': ('There are more important things than environment', down),
                 'v202': ('Others should start to protect the environment', down),
                 'v203': ('Environmental threats are exaggerated', down),
                'v149':	('Do you justify: claiming state benefits', UP),
                'v159':	('Do you justify: avoiding a fare on public transport', UP),
                'v150':	('Do you justify: cheating on tax', UP),
                'v152':	('Do you justify: accepting a bribe', UP),
                'v153':	('Do you justify: homosexuality', UP),
                'v160':	('Do you justify: prostitution', UP),
                'v154':	('Do you justify: abortion', UP),
                'v155':	('Do you justify: divorce', UP),
                'v156':	('Do you justify: euthanasia', UP),
                'v157':	('Do you justify: suicide', UP),
                'v158': ('Do you justify: having casual sex', UP),
                'v163':	('Do you justify: death penalty', UP)}

    marker    = ['v275b_N2', 'c_abrv', 'v275b_N1']

    def __init__(self, min_entries=40, max_entries=4000):
        
        questions = list(self.legend.keys())
            
        df        = pd.read_stata("Datasets/EVS2020/EVS.dta", 
                                   convert_categoricals=False,
                                   columns = questions+self.marker)
        
        print(f"Loaded {len(df)} questionaires")
        df[df[questions]<0] = np.NaN
        df.dropna(inplace=True)
        print(f"Kept {len(df)} non-empty questionaires")
    
        ### germany has different structure
        de = df.loc[df.c_abrv == 'DE']
        de.set_index('v275b_N1', inplace=True, drop=True)
        de = de.drop('v275b_N2', axis=1)
          
        ### merging germany with rest      
        df.set_index('v275b_N2', inplace=True, drop=True)
        df.drop(['-4','-1'], inplace=True)
        df.drop('v275b_N1', axis=1, inplace=True)
        
        ### labels
        data = pd.concat([de,df])
        self.labels = data.c_abrv.map(str.lower).to_dict()
        data.drop('c_abrv', axis=1, inplace=True)
        
        sizes = data.groupby(level=0).size() 
        
        for key in questions:
            bottom, top    = self.legend[key][1]
            minval, maxval = self.interval
            data[key] = np.round(minval + (data[key]-bottom)*(maxval-minval)/(top-bottom), 2)
            
        self.data = data.loc[ (min_entries < sizes) & (sizes < max_entries)]
    
    def labeldict(self):
        return self.labels 
        
if __name__ == '__main__':
    EVS = EuropeanValueStudy()
    data = EVS.data
    # labels = EVS.labels
