# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:36:25 2017

Analyze Trajectory Results

@author: nsgat
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cwd = os.getcwd()

all_folders = os.listdir('./Data/')

#folders = ['hale_2017_09_15_14_16_06', # Linear solver comparison
#           'hale_2017_09_15_14_16_27',
#           'hale_2017_09_15_14_16_40',
#           'hale_2017_09_15_14_16_56']
#
#solver = ['ma27','ma57','ma77','ma97']


#folders = ['hale_2017_10_21_23_59_12', # 9 optimization case comparison
#           'hale_2017_10_22_00_00_13',
#           'hale_2017_10_22_00_00_47',
#           'hale_2017_10_22_00_01_20',
#           'hale_2017_10_22_01_01_26',
#           'hale_2017_10_22_01_02_37',
#           'hale_2017_10_22_01_04_14',
#           'hale_2017_10_22_01_04_42',
#           'hale_2017_10_22_01_16_35']
#
#solver = ['130 kg Batt',
#          '135 kg Batt',
#          '140 kg Batt',
#          '145 kg Batt',
#          '15 min horizon',
#          '10 min horizon',
#          '8 min horizon',
#          '5 min horizon',
#          '20 sec timesetp'] # Identifier

# Logan's workstation time test           
folders = ['hale_2017_10_31_17_14_51 - Benchmark byu ma27 Batt 139.5 kg', # 9 optimization case comparison
           'hale_2017_10_31_17_19_28 - Benchmark Logan ma27 Batt 139.5 kg',
           'hale_2017_10_31_17_38_29 - Benchmark xps ma27 Batt 139.5 kg']

solver = ['byu ma27',
          'Logan ma27',
          'xps ma27'] # Identifier          

n = len(folders)
name = []
data = {}
for i in range(n):
    for j in range(len(all_folders)):
        folder_key = folders[i] in os.listdir('./Data/')[j]
        if folder_key == True:
            key_idx = j
    name.append(os.listdir('./Data/' + all_folders[key_idx] + '/Intermediates')[-1]) # Assumes one open file per folder (does it still?)
    data[i] = pd.read_excel('./Data/' + all_folders[key_idx] + '/Intermediates/' + name[i])

#%%

max_len = len(data[0])
max_idx = 0
for i in range(1,n):
    idx = len(data[i])
    if idx > max_idx:
        max_len = idx
        max_idx = i
        
#order = [3,0,1,2]
order = range(n)

time = pd.DataFrame()
for i in order:
    time[i] = data[i]['iteration_time']
    for j in range(2,len(data[i]['iteration_time'])):
        time[i][j] = data[i]['iteration_time'][j] + time[i][j-1]

#%% Logan's workstation time test
# Analyze time[1] case

time_analyzed = {}
time_analyzed[0] = 1-time[1]/time[2]
time_analyzed[1] = 1-time[1]/time[0]

time_len = time[1].last_valid_index()

plt.figure()
plt.plot(data[1]['time'][1:time_len],time_analyzed[0][1:time_len], label = 'xps')
plt.plot(data[1]['time'][1:time_len],time_analyzed[1][1:time_len], label = 'byu')
plt.title('Optimization Time: Logan faster than others')
plt.legend()

np.average(time_analyzed[0][1:2414])*100
np.average(time_analyzed[1][1:2413])*100

   
#%%
        
resolve = pd.DataFrame()
for i in order:
    resolve[i] = data[i]['re-solve']
    for j in range(2,len(data[i]['re-solve'])):
        resolve[i][j] = data[i]['re-solve'][j] + resolve[i][j-1]


#%%
        
success = pd.DataFrame()
for i in order:
    success[i] = 1 - data[i]['successful_solution']
    for j in range(2,len(data[i]['successful_solution'])):
        success[i][j] = 1 - data[i]['successful_solution'][j] + success[i][j-1]
        
        
#%%

plt.figure()
for i in range(n):
    l_time = len(data[i]['time'])
    l_cum_time = len(time[i])
    if l_time < l_cum_time:
        l_min = l_time
    else:
        l_min = l_cum_time
    plt.plot(data[i]['time'][0:l_min]/3600,time[i][0:l_min]/3600, label=solver[i])
plt.ylabel('Iteration Time (hr)')
plt.xlabel('Horizon Time (hr)')
#plt.xlim([0, 600])
#plt.ylim([0, 3000])
plt.legend()


plt.figure()
for i in range(n):
    l_time = len(data[i]['time'])
    l_cum_time = len(time[i])
    if l_time < l_cum_time:
        l_min = l_time
    else:
        l_min = l_cum_time
    plt.plot(data[i]['time'][0:l_min]/3600,data[i]['te'][0:l_min], label=solver[i])
plt.ylabel('Total Energy (MJ)')
plt.xlabel('Horizon Time (hr)')
plt.legend()


#plt.figure()
#for i in range(n):
#    l_time = len(data[i]['time'])
#    l_cum_time = len(time[i])
#    if l_time < l_cum_time:
#        l_min = l_time
#    else:
#        l_min = l_cum_time
#    plt.loglog(data[i]['time'][0:l_min]/3600,resolve[i][0:l_min], label=solver[i])
#plt.ylabel('Number of Re-solves')
#plt.xlabel('Horizon Time (hr)')
#plt.legend()


plt.figure()
for i in range(n):
    l_time = len(data[i]['time'])
    l_cum_time = len(time[i])
    if l_time < l_cum_time:
        l_min = l_time
    else:
        l_min = l_cum_time
    plt.semilogy(data[i]['time'][0:l_min]/3600,success[i][0:l_min], label=solver[i])
plt.ylabel('Number of Unsuccessful Solutions')
plt.xlabel('Horizon Time (hr)')
plt.legend()


sum(success - resolve)

plt.figure()



