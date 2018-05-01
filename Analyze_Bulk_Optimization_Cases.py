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
import time
import glob
import yaml

cwd = os.getcwd()

all_folders = os.listdir('./Data/')


#folders = ['hale_2017_10_31_17_14_51 - Benchmark byu ma27 Batt 139.5 kg', # 9 optimization case comparison
#           'hale_2017_10_31_17_19_28 - Benchmark Logan ma27 Batt 139.5 kg',
#           'hale_2017_10_31_17_38_29 - Benchmark xps ma27 Batt 139.5 kg']
#
#solver = ['byu ma27',
#          'Logan ma27',
#          'xps ma27'] # Identifier          


folders = all_folders.copy()
           

n = len(folders)
name = {}
info = {} # 1=success, 0=error
data = {}
yaml_name = {}
config = {}
start = time.time()
i = 21
for i in range(n):
    for j in range(len(all_folders)):
        folder_key = folders[i] in os.listdir('./Data/')[j]
        if folder_key == True:
            key_idx = j
    try:
        name[i] = (os.listdir('./Data/' + all_folders[key_idx] + '/Intermediates')[-1])
        data[i] = pd.read_excel('./Data/' + all_folders[key_idx] + '/Intermediates/' + name[i])
        info[i] = 1 # Success
        
        for file in os.listdir('./Data/' + all_folders[key_idx]):
            if file.endswith('.yml'):
                yaml_name[i] = file
#        yaml_name[i] = glob.glob('.\\Data\\' + all_folders[key_idx] + '\\*.yml') 
        with open('./Data/' + all_folders[key_idx] + '/' + yaml_name[i], 'r') as stream:
            try:
                config[i] = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        info[i] = 0 # Error
    print("Loaded " + str(i+1) + " of " + str(len(all_folders)+1) + " (" + "{:.2f}".format(time.time() - start) + " sec)")
end = time.time()
print("Time: " + "{:.2f}".format(end - start) + " seconds")


#%%

plt.figure()
for i in range(1,len(folders)):
    if info[i] == 1:
        if data[i]['te'][0] < 40:
            plt.plot(data[i]['time']/3600, data[i]['te'], label = folders[i])
plt.legend()

#%%

plt.figure()
files = []
max_te = []
max_te_time = []
end_time = []
for i in range(1,len(folders)):
    if info[i] == 1:
        if data[i]['te'][0] < 40 and data[i]['time'].iloc[-1]/3600 > 6 and folders[i].find("Wind") == -1 :
            plt.plot(data[i]['time']/3600, data[i]['te'], label = folders[i])
            plt.text(data[i]['time'].iloc[-1]/3600, data[i]['te'].iloc[-1], folders[i])
            files.append(folders[i])
            max_te.append(max(data[i]['te']))
            max_te_time.append(data[i]['time'][data[i]['te'][data[i]['te'] == max_te[-1]].index.tolist()[0]]/3600)
            end_time.append(data[i]['time'].iloc[-1]/3600)
#plt.legend()

#%%
analysis = pd.DataFrame(np.c_[files, max_te, max_te_time, end_time], columns = ['filename', 'max_te', 'max_te_time', 'end_time'])

pd.concat([files, max_te, max_te_time, end_time], axis = 1)


analysis = pd.DataFrame(
#    {'filename': files,
    {'max_te': max_te,
     'max_te_time': max_te_time,
     'end_time': end_time
    })
    

df = pd.to_numeric(analysis[['max_te_time', 'max_te']])
df.plot()

#%%
plt.figure()
x = analysis['max_te_time'] # np.array(analysis['max_te_time'].astype(float)) # pd.to_numeric(analysis['max_te_time'])
y = analysis['max_te'] # np.array(analysis['max_te'].astype(float)) # pd.to_numeric(analysis['max_te'])
text = files #text = analysis['filename'].to_string()
plt.scatter(x, y)
for i in range(len(text)):
    plt.text(x[i], y[i], text[i])

plt.text(x, y, text)#(convert_string=True))

#%%    
type(x)

analysis['max_te'].astype(float)

#%%

match = [i for i in range(0,len(data[i]['time'])) if data[i]['te'].iloc[i] == max_te[-1]]

data[i]['te'].index(max_te[-1])


data[i]['te'][data[i]['te'] == max_te[-1]].index.tolist()

test = list(np.where(data[i]['te']==max_te[-1])[0])[0]

np.where(data[i]['te']==max_te[-1])[0][0]

data[i]['te'][test]
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



