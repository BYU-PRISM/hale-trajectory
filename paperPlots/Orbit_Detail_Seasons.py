# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 21:11:25 2018

@author: Nathaniel Gates
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

#winter_folder = 'hale_2018_07_01_00_07_38 - Winter Timestep 8 sec Horizon 15 min CL 1.5'
#spring_folder = 'hale_2018_07_01_00_18_43 - Spring Timestep 8 sec Horizon 15 min CL 1.5'
#summer_folder = 'hale_2018_07_01_00_19_26 - Summer Timestep 8 sec Horizon 15 min CL 1.5'
#fall_folder = 'hale_2018_07_01_00_20_06 - Fall Timestep 8 sec Horizon 15 min CL 1.5'

# Used in August
winter_folder = 'hale_2018_08_27_12_16_00 - Winter Battery 136'
spring_folder = 'hale_2018_08_27_12_16_10 - Spring Battery 136'
summer_folder = 'hale_2018_08_27_12_16_19 - Summer Battery 136'
fall_folder = 'hale_2018_08_27_12_16_26 - Fall Battery 136'

save_plots = True

data_path = '../Data/'
#file = glob.glob(data_path+spring_folder+'/opt_results*.xlsx')[0]
file = glob.glob(data_path+spring_folder+'/Intermediates/*.xlsx')[0]

d = pd.read_excel(file)



#%%

plt.style.use(['seaborn-paper','seaborn-whitegrid'])
plt.rc("font", family="serif")
#plt.rc('text', usetex=False)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
width  = 6
height = width / 1.618


#%% Plot all at once 3x3
#   Plot 1/4

# Plot a single orbit
start = 900+7 # int(120*60/15)
length = 59 # int(9*60/15)

# Used in August
start = 900+13 # int(120*60/15)
length = 56 # int(9*60/15)

end = start + length
print(d.time[start:end].reset_index(drop=True)[0]/60)

xpts = np.arange(1,len(d.h[start:end])+1)*8/60 # This is now time in min

plt.figure(figsize=(5,5))
#plt.title('Complete Orbit (2 hrs after dawn)')
y = d.x[start:end].reset_index(drop=True)/1000 # Flipped to match a map
x = d.y[start:end].reset_index(drop=True)/1000
plt.plot(x, y, color = 'C0', linestyle = '-', linewidth = 1, alpha=0.5)
plt.scatter(x, y, c=xpts, cmap=plt.cm.viridis)
plt.xlim([-3,3])
plt.ylim([-3,3])
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)

circle1 = plt.Circle((0, 0), 3, color='k', fill=False, alpha=0.5, linestyle='--', linewidth=2)
ax.add_artist(circle1)

azimuth = (d.azimuth[start:end].mean()) * np.pi / 180
#plt.plot([0,1*np.cos(azimuth)], [0,1*np.sin(azimuth)], color = 'C1')
plt.plot([0,1*np.sin(azimuth)], [0,1*np.cos(azimuth)], color = 'C4') # Switched directions
plt.scatter(0,0, s = 100, color = 'C4')

plt.ylabel('N (km)')
plt.xlabel('E (km)')

if save_plots == True:
    plt.savefig('spring_orbit.pdf', bbox_inches='tight')

#%%

#file = glob.glob(data_path+summer_folder+'/opt_results*.xlsx')[0]
file = glob.glob(data_path+summer_folder+'/Intermediates/*.xlsx')[0]

d = pd.read_excel(file)

# Plot a single orbit
#start = 900+7 # int(120*60/15)
#length = 59 # int(9*60/15)
#end = start + length
print(d.time[start:end].reset_index(drop=True)[0]/60)

xpts = np.arange(1,len(d.h[start:end])+1)*8/60 # This is now time in min

plt.figure(figsize=(5,5))
#plt.title('Complete Orbit (2 hrs after dawn)')
y = d.x[start:end].reset_index(drop=True)/1000 # Flipped to match a map
x = d.y[start:end].reset_index(drop=True)/1000
plt.plot(x, y, color = 'C0', linestyle = '-', linewidth = 1, alpha=0.5)
plt.scatter(x, y, c=xpts, cmap=plt.cm.viridis)
plt.xlim([-3,3])
plt.ylim([-3,3])
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)

circle1 = plt.Circle((0, 0), 3, color='k', fill=False, alpha=0.5, linestyle='--', linewidth=2)
ax.add_artist(circle1)

azimuth = (d.azimuth[start:end].mean()) * np.pi / 180
#plt.plot([0,1*np.cos(azimuth)], [0,1*np.sin(azimuth)], color = 'C1')
plt.plot([0,1*np.sin(azimuth)], [0,1*np.cos(azimuth)], color = 'C4') # Switched directions
plt.scatter(0,0, s = 100, color = 'C4')

plt.ylabel('N (km)')
plt.xlabel('E (km)')

if save_plots == True:
    plt.savefig('summer_orbit.pdf', bbox_inches='tight')
    
#%%

#file = glob.glob(data_path+fall_folder+'/opt_results*.xlsx')[0]
file = glob.glob(data_path+fall_folder+'/Intermediates/*.xlsx')[0]

d = pd.read_excel(file)

# Plot a single orbit
#start = 900+7 # int(120*60/15)
#length = 59 # int(9*60/15)
#end = start + length
print(d.time[start:end].reset_index(drop=True)[0]/60)

xpts = np.arange(1,len(d.h[start:end])+1)*8/60 # This is now time in min

plt.figure(figsize=(5,5))
#plt.title('Complete Orbit (2 hrs after dawn)')
y = d.x[start:end].reset_index(drop=True)/1000 # Flipped to match a map
x = d.y[start:end].reset_index(drop=True)/1000
plt.plot(x, y, color = 'C0', linestyle = '-', linewidth = 1, alpha=0.5)
plt.scatter(x, y, c=xpts, cmap=plt.cm.viridis)
plt.xlim([-3,3])
plt.ylim([-3,3])
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)

circle1 = plt.Circle((0, 0), 3, color='k', fill=False, alpha=0.5, linestyle='--', linewidth=2)
ax.add_artist(circle1)

azimuth = (d.azimuth[start:end].mean()) * np.pi / 180
#plt.plot([0,1*np.cos(azimuth)], [0,1*np.sin(azimuth)], color = 'C1')
plt.plot([0,1*np.sin(azimuth)], [0,1*np.cos(azimuth)], color = 'C4') # Switched directions
plt.scatter(0,0, s = 100, color = 'C4')

plt.ylabel('N (km)')
plt.xlabel('E (km)')

if save_plots == True:
    plt.savefig('fall_orbit.pdf', bbox_inches='tight')
    
#%%

#file = glob.glob(data_path+winter_folder+'/opt_results*.xlsx')[0]
file = glob.glob(data_path+winter_folder+'/Intermediates/*.xlsx')[0]

d = pd.read_excel(file)

# Plot a single orbit
#start = 900+7 # int(120*60/15)
#length = 59 # int(9*60/15)
#end = start + length
print(d.time[start:end].reset_index(drop=True)[0]/60)

xpts = np.arange(1,len(d.h[start:end])+1)*8/60 # This is now time in min

plt.figure(figsize=(5,5))
#plt.title('Complete Orbit (2 hrs after dawn)')
y = d.x[start:end].reset_index(drop=True)/1000 # Flipped to match a map
x = d.y[start:end].reset_index(drop=True)/1000
plt.plot(x, y, color = 'C0', linestyle = '-', linewidth = 1, alpha=0.5)
plt.scatter(x, y, c=xpts, cmap=plt.cm.viridis)
plt.xlim([-3,3])
plt.ylim([-3,3])
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)

circle1 = plt.Circle((0, 0), 3, color='k', fill=False, alpha=0.5, linestyle='--', linewidth=2)
ax.add_artist(circle1)

azimuth = (d.azimuth[start:end].mean()) * np.pi / 180
#plt.plot([0,1*np.cos(azimuth)], [0,1*np.sin(azimuth)], color = 'C1')
plt.plot([0,1*np.sin(azimuth)], [0,1*np.cos(azimuth)], color = 'C4') # Switched directions
plt.scatter(0,0, s = 100, color = 'C4')

plt.ylabel('N (km)')
plt.xlabel('E (km)')

if save_plots == True:
    plt.savefig('winter_orbit_unlabeled.pdf', bbox_inches='tight')