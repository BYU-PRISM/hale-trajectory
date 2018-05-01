# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:47:07 2017

Analyze total energy of different timeshift cases

@author: nsgat
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cwd = os.getcwd()

all_folders = os.listdir('./Data/')

## Logan's workstation time test           
#folders = ['hale_2017_11_01_13_06_28 - 15sec 10min 4shift xps ma57 Batt 139.5kg',
#           'hale_2017_10_31_17_19_28 - Benchmark Logan ma27 Batt 139.5 kg',
#           'hale_2017_10_31_17_38_29 - Benchmark xps ma27 Batt 139.5 kg']
#
#solver = ['byu ma27',
#          'Logan ma27',
#          'xps ma27'] # Identifier          

folders = ['hale_2017_11_01_16_03_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg Orbit 6 km',
           'hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg']

solver = ['SS Climb',
          'Opt'] # Identifier
          
#n = len(folders)
#name = []
#data = {}
#for i in range(n):
#    for j in range(len(all_folders)):
#        folder_key = folders[i] in os.listdir('./Data/')[j]
#        if folder_key == True:
#            key_idx = j
#    name.append(os.listdir('./Data/' + all_folders[key_idx] + '/Intermediates')[-1]) # Assumes one open file per folder (does it still?)
#    data[i] = pd.read_excel('./Data/' + all_folders[key_idx] + '/Intermediates/' + name[i])

# Import files by hand
data = {}
#data[0] = pd.read_excel('./Data/hale_2017_11_01_16_03_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg Orbit 6 km/ss_results_2017_11_01_16_03_24_test.xlsx')
#data[1] = pd.read_excel('./Data/hale_2017_11_01_16_03_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg Orbit 6 km/Intermediates/iter_4560.xlsx')
#data[2] = pd.read_excel('./Data/hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg/ss_results_2017_11_01_13_28_24_test.xlsx')
#data[3] = pd.read_excel('./Data/hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg/Intermediates/iter_5710.xlsx')


#%%
# Plot total energy
plt.figure()
plt.plot(data[0]['time']/3600, data[0]['te']/3.6, label = 'SS Circle')
#plt.plot(data[1]['time']/3600, data[1]['te']/3.6, label = 'Opt Circle')
#plt.plot(data[2]['time']/3600, data[2]['te']/3.6, label = 'SS')
plt.plot(data[3]['time']/3600, data[3]['te']/3.6, label = 'Opt')
plt.ylabel('Total Energy (kWh)')
plt.xlabel('Time (hr)')
plt.xlim([0, 24])
plt.legend()


#%%

plt.plot(data[1]['te']/data[3]['te'])


#%%

def find_nearest(array1,value,over_time_index='none'):
    if type(over_time_index) is not str:
        over_time_index=int(over_time_index)
        idx = (np.abs(np.array(array1)[over_time_index:]-value)).argmin()
        return array1[idx+over_time_index], (idx+over_time_index)
    else: 
        idx = (np.abs(np.array(array1)[:]-value)).argmin()
        return array1[idx], (idx)


#%%

# Create custom plots of total energy

# Root directory = ~/hale-optimization/Trajectory/
data = {}
#data[0] = pd.read_excel('./Data/hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg/ss_results_2017_11_01_13_28_24_test.xlsx')
#data[1] = pd.read_excel('./Data/hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg/Intermediates/iter_5710.xlsx')

data[0] = pd.read_excel('./Data/hale_2017_11_01_13_06_28 - 15sec 10min 4shift xps ma57 Batt 139.5kg/ss_results_2017_11_01_13_06_28_test.xlsx')
data[1] = pd.read_excel('./Data/hale_2017_11_01_13_06_28 - 15sec 10min 4shift xps ma57 Batt 139.5kg/Intermediates/iter_5716.xlsx')


g = 9.80665 # Gravity (m/s^2)
battery_mass = 139.5 # kg
total_mass = 352.5 # kg
energy_density = 350 # W-hr/kg
max_battery_energy = battery_mass*energy_density/1000 # kW-hr
max_delta_h_ft = 90000 - 60000
max_delta_h_m = max_delta_h_ft/3.28084
max_potential_energy_Nm = total_mass*g*max_delta_h_m
max_potential_energy_kWhr = max_potential_energy_Nm/3600000
max_energy = max_battery_energy + max_potential_energy_kWhr

SOC_min = 0.20
energy_min = SOC_min*max_battery_energy

#%%
# Plot total energy
plt.figure()
plt.plot(data[0]['time']/3600, data[0]['te']/3.6, label = 'SS')
plt.plot(data[1]['time']/3600, data[1]['te']/3.6, label = 'Opt')
plt.plot(data[1]['time']/3600, np.ones(len(data[1]['time']))*energy_min, 'k--')
plt.plot(data[1]['time']/3600, np.ones(len(data[1]['time']))*max_energy, 'k--')
plt.ylabel('Total Energy (kWh)')
plt.xlabel('Time (hr) Since Dawn (7:15 AM) on Dec 21, 2016')
plt.xlim([0, 24])
plt.ylim([0, 60])
plt.xticks([0,4,8,12,16,20,24])
#plt.legend(loc=1)
plt.legend(bbox_to_anchor=(0.79,0.94),loc=2,ncol=1,fontsize=10,markerfirst=False,framealpha=1.0)
plt.minorticks_on()


#%%
ss_time = data[0]['time']/3600
opt_time = data[1]['time']/3600

ss_potential = (data[0]['h']-18288)*g*total_mass/3600000
opt_potential = (data[1]['h']-18288)*g*total_mass/3600000

ss_soc = data[0]['e_batt']/data[0]['e_batt_max'][0]*100
opt_soc = data[1]['e_batt']/data[0]['e_batt_max'][0]*100

plt.figure()
plt.subplot(2,1,1)
plt.plot(ss_time,ss_soc, label = 'SS')#, linewidth = 2)
plt.plot(opt_time,opt_soc, label = 'Opt')#, linewidth = 2)
plt.ylim([-10, 110])
plt.xlim([0, 24])
plt.ylabel('Battery SOC (%)')
plt.xticks([0,4,8,12,16,20,24])
plt.xlabel('Time (hr) Since Dawn (7:15 AM) on Dec 21, 2016')
plt.minorticks_on()
ax = plt.gca()
ax.tick_params(axis='y',which='minor',left='off', right='off')
plt.legend(bbox_to_anchor=(0.79,0.94),loc=2,ncol=1,fontsize=10,markerfirst=False,framealpha=1.0)



plt.subplot(2,1,2)
plt.plot(ss_time,data[0]['h']/1000, label = 'SS')#, linewidth = 2)
plt.plot(opt_time,data[1]['h']/1000, label = 'Opt')#, linewidth = 2)
plt.legend(bbox_to_anchor=(0.79,0.94),loc=2,ncol=1,fontsize=10,markerfirst=False,framealpha=1.0)
plt.plot(ss_time,np.ones(len(ss_time))*90000/3.28084/1000, 'k--')
plt.plot(ss_time,np.ones(len(ss_time))*60000/3.28084/1000, 'k--')


#plt.ylim([-5, 105])
plt.xlim([0, 24])
plt.ylim([17,28])
plt.ylabel('Altitude (km)')
plt.xticks([0,4,8,12,16,20,24])
plt.xlabel('Time (hr) Since Dawn (7:15 AM) on Dec 21, 2016')
plt.minorticks_on()
ax = plt.gca()
ax.tick_params(axis='y',which='minor',left='off', right='off')




#%%

find_nearest(data[0]['te']/3.6, energy_min, over_time_index=1000)

[minval, minindex] = find_nearest(data[0]['te']/3.6,energy_min)
time_empty = round(data[0]['te'][minindex],2)

plt.axvline(x=time_empty)

#%%


plt.plot(np.ones(len(data[1]['time']))*time_empty,minval)


[xmin, xmax, ymin, ymax] = plt.axis()
plt.text(xmax*0.98,ymax*0.8,"Battery Empty Times:",ha='right',fontsize=7)

battery_empty_line = np.ones(len(data[1]['time']))*soc_empty*simulation_results[0].e_batt_tot
plt.plot(Time,battery_empty_line,'k--')
plt.axvline(x=sunset_time, ymin=-1, ymax = 1, linewidth=1, color='r')
plt.text(sunset_time-0.75,ymax-0.75*(ymax-ymin),'Sunset    ' + str("%.2f" % sunset_time),rotation=90,fontsize=9)
plt.legend(bbox_to_anchor=(0,0.985),loc=2,ncol=1,fontsize=7,markerfirst=False,framealpha=1.0)



#%%


#
##%%
#
#max_len = len(data[0])
#max_idx = 0
#for i in range(1,n):
#    idx = len(data[i])
#    if idx > max_idx:
#        max_len = idx
#        max_idx = i
#        
##order = [3,0,1,2]
#order = range(n)
#
#time = pd.DataFrame()
#for i in order:
#    time[i] = data[i]['iteration_time']
#    for j in range(2,len(data[i]['iteration_time'])):
#        time[i][j] = data[i]['iteration_time'][j] + time[i][j-1]
#
##%% Logan's workstation time test
## Analyze time[1] case
#
#time_analyzed = {}
#time_analyzed[0] = 1-time[1]/time[2]
#time_analyzed[1] = 1-time[1]/time[0]
#
#time_len = time[1].last_valid_index()
#
#plt.figure()
#plt.plot(data[1]['time'][1:time_len],time_analyzed[0][1:time_len], label = 'xps')
#plt.plot(data[1]['time'][1:time_len],time_analyzed[1][1:time_len], label = 'byu')
#plt.title('Optimization Time: Logan faster than others')
#plt.legend()
#
#np.average(time_analyzed[0][1:2414])*100
#np.average(time_analyzed[1][1:2413])*100
#
#   
##%%
#        
#resolve = pd.DataFrame()
#for i in order:
#    resolve[i] = data[i]['re-solve']
#    for j in range(2,len(data[i]['re-solve'])):
#        resolve[i][j] = data[i]['re-solve'][j] + resolve[i][j-1]
#
#
##%%
#        
#success = pd.DataFrame()
#for i in order:
#    success[i] = 1 - data[i]['successful_solution']
#    for j in range(2,len(data[i]['successful_solution'])):
#        success[i][j] = 1 - data[i]['successful_solution'][j] + success[i][j-1]
#        
#        
##%%
#
#plt.figure()
#for i in range(n):
#    l_time = len(data[i]['time'])
#    l_cum_time = len(time[i])
#    if l_time < l_cum_time:
#        l_min = l_time
#    else:
#        l_min = l_cum_time
#    plt.plot(data[i]['time'][0:l_min]/3600,time[i][0:l_min]/3600, label=solver[i])
#plt.ylabel('Iteration Time (hr)')
#plt.xlabel('Horizon Time (hr)')
##plt.xlim([0, 600])
##plt.ylim([0, 3000])
#plt.legend()
#
#
#plt.figure()
#for i in range(n):
#    l_time = len(data[i]['time'])
#    l_cum_time = len(time[i])
#    if l_time < l_cum_time:
#        l_min = l_time
#    else:
#        l_min = l_cum_time
#    plt.plot(data[i]['time'][0:l_min]/3600,data[i]['te'][0:l_min], label=solver[i])
#plt.ylabel('Total Energy (MJ)')
#plt.xlabel('Horizon Time (hr)')
#plt.legend()
#
#
##plt.figure()
##for i in range(n):
##    l_time = len(data[i]['time'])
##    l_cum_time = len(time[i])
##    if l_time < l_cum_time:
##        l_min = l_time
##    else:
##        l_min = l_cum_time
##    plt.loglog(data[i]['time'][0:l_min]/3600,resolve[i][0:l_min], label=solver[i])
##plt.ylabel('Number of Re-solves')
##plt.xlabel('Horizon Time (hr)')
##plt.legend()
#
#
#plt.figure()
#for i in range(n):
#    l_time = len(data[i]['time'])
#    l_cum_time = len(time[i])
#    if l_time < l_cum_time:
#        l_min = l_time
#    else:
#        l_min = l_cum_time
#    plt.semilogy(data[i]['time'][0:l_min]/3600,success[i][0:l_min], label=solver[i])
#plt.ylabel('Number of Unsuccessful Solutions')
#plt.xlabel('Horizon Time (hr)')
#plt.legend()
#
#
#sum(success - resolve)
#
#plt.figure()