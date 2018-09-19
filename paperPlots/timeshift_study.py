# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:25:04 2018

@author: PRISM Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

cwd = os.getcwd()

data_path = '../Data/'

files = os.listdir(data_path)
folders = ['hale_2018_08_23_14_18_05 - Winter Timestep 8 sec Timeshift 16 sec',
           'hale_2018_08_23_14_21_27 - Winter Timestep 8 sec Timeshift 32 sec',
           'hale_2018_08_23_14_22_10 - Winter Timestep 8 sec Timeshift 64 sec',
           'hale_2018_08_23_14_24_16 - Winter Timestep 8 sec Timeshift 128 sec',
           'hale_2018_08_23_17_46_16 - Winter Timestep 8 sec Timeshift 192 sec',
           'hale_2018_08_23_14_53_59 - Winter Timestep 8 sec Timeshift 256 sec',
           'hale_2018_08_23_17_47_26 - Winter Timestep 8 sec Timeshift 384 sec']
#           'hale_2018_08_23_15_13_54 - Winter Timestep 8 sec Timeshift 512 sec']

timeshift = np.array([16,32,64,128,192,256,384])#,512])

##%%
#d = {}
#i=-1
#for folder in folders:
#    i+=1
#    print('Reading folder '+str(i+1)+' of '+str(len(folders)))
#    file = glob.glob('../Data/'+folders[i]+'/opt*') [0]   
#    d[i] = pd.read_excel(file)


#%% Grab completed optimization data (from opt_results)
file = {}
for i in range(len(folders)):
    try:
        file[i] = glob.glob(data_path+folders[i]+'/opt_results*.xlsx')[0]
    except:
        file[i] = ''

ss_file = {}
for i in range(len(folders)):
    try:
        ss_file[i] = glob.glob(data_path+folders[i]+'/ss_results*.xlsx')[0]
    except:
        ss_file[i] = ''        

##%% Grab partially completed optimization data (from intermediates)
#file = {}
#for i in range(len(folders)):
#    try:
#        file[i] = glob.glob(data_path+folders[i]+'/intermediates/iter*.xlsx')[-1]
#    except:
#        file[i] = ''

# Read in the data
d = {}
for i in range(len(file)):
    print('Reading Opt File '+str(i+1)+' of '+str(len(file)))
    try:
        d[i] = pd.read_excel(file[i])
    except:
        print('Skipped opt file '+str(i+1))  

ss = {}
for i in range(len(file)): 
    print('Reading SS File '+str(i+1)+' of '+str(len(file)))
    try:
        ss[i] = pd.read_excel(ss_file[i])
    except:
        print('Skipped ss file '+str(i+1))  


#%% Import large excel files

## Read the file
#data = pd.read_excel(file[i], low_memory=False)
#
## Output the number of rows
#print("Total rows: {0}".format(len(data)))

# Configure plotting
plt.style.use(['seaborn-paper','seaborn-whitegrid'])
plt.rc("font", family="serif")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)



    
#%%
n = len(folders)
plt.figure()
for i in range(n):
    plt.plot(d[i].time/3600, d[i].te/3.6, label = str(i+1))
    
#%%
e_final = np.zeros(n)
for i in range(n):
    e_final[i] = d[i]['te'].iloc[-1]/3.6

plt.figure()
for i in range(n):
    plt.scatter(i+1,e_final[i])


#%%
plt.figure(figsize=(7,10))
for i in range(len(d)):
    plt.subplot(len(d),1,i+1)
    plt.plot(d[i].time/3600, d[i].iteration_time)
    plt.ylabel(timeshift[i])
#    plt.ylim(0,500)
    plt.xlim(0,24)
plt.tight_layout()

print('\nAvg Iteration Time Before Sunset:')
for i in range(len(d)):
    print('{:.0f}'.format((timeshift[i]/8))+' Timestep Timeshift: {:.2f} sec'.format(d[i].iteration_time[0:450].mean()))

print('\nAvg Iteration Time After Sunset:')
for i in range(len(d)):
    print('{:.0f}'.format((timeshift[i]/8))+' Timestep Timeshift: {:.2f} sec'.format(d[i].iteration_time[450:].mean()))

print('\nAvg Iteration Time Ratio Before and After Sunset:')
for i in range(len(d)):
    print('{:.0f}'.format((timeshift[i]/8))+' Timestep Timeshift: {:.2f} times'.format(d[i].iteration_time[0:450].mean()/d[i].iteration_time[450:].mean()))
