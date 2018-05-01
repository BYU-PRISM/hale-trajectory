# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:35:05 2017

@author: NathanielGates
"""

import os
import pandas as pd
import numpy as np

files = os.listdir('./Data')


d = {}
name = {}
idx = np.zeros([len(files)])
for i in range(1,len(files)):
    name[i] = os.listdir('./Data/'+files[i])  
    
    for j in range(len(name[i])):
        if 'opt_results' in name[i][j]:
            idx[i] = j
    
#    name[i][int(idx[i])] #] == 'opt_results'
    

#%%

for i in range(len(files)):
    if int(idx[i]) != 0:
        d[i] = pd.read_excel('./Data/'+files[i]+'/'+name[i][int(idx[i])])

#name[i][int(idx[i])]

1 != 1
