# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:41:35 2017

@author: nsgat
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()

path = r'.\Data\hale_2017_11_07_18_06_13 - Double alpha dcost\thermal_data'
os.chdir(path)

file = os.listdir()

#%%

data = {}
for i in range(len(file)):
    data[i] = pd.read_csv('./' + file[i])

#data[i].reindex_axis(sorted(data[i].columns), axis=1)
#%%

plt.figure()
plt.plot(data[2]['time']/3600, data[2]['t_3[1]']-273.15)
plt.plot(data[2]['time']/3600, data[2]['t_3[20]']-273.15)

#%%

plt.figure()
plt.plot(data[2]['time']/3600, data[2]['t_3[1]']-273.15)
plt.plot(data[2]['time']/3600, data[2]['t_3[20]']-273.15)
