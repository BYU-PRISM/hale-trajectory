# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 21:11:25 2018

@author: Nathaniel Gates
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.integrate as integrate

folder = 'hale_2018_06_30_02_42_03 - Winter Timestep 8 sec Horizon 15 min'
#folder = 'hale_2018_06_30_02_44_31 - Winter Timestep 8 sec Horizon 17.5 min'
#folder = 'hale_2018_02_23_14_41_32 - E216 Winter Density'
folder = 'hale_2018_06_30_02_42_03 - Winter Timestep 8 sec Horizon 15 min' # Case 1

#case = 'max_te'
case = 'max_ebatt'

if case == 'max_te':
#    folder = 'hale_2018_07_01_00_07_38 - Winter Timestep 8 sec Horizon 15 min CL 1.5' # Case 2
    folder = 'hale_2018_08_27_12_16_00 - Winter Battery 136'
    name = 'max_te'
    print('Case selected: '+str(folder))
elif case == 'max_ebatt':
#    folder = 'hale_2018_07_01_01_10_20 - Winter Timestep 8 sec Horizon 15 min Max Batt' # Case 3
    folder = 'hale_2018_08_27_13_01_10 - Winter Maximize Battery'
    name = 'max_ebatt'
    print('Case selected: '+str(folder))    
else:
    print('No case defined.')

save_plots = False

data_path = '../Data/'
#file = glob.glob(data_path+folder+'/opt_results*.xlsx')[0]
file = glob.glob(data_path+folder+'/intermediates/*.xlsx')[0]


d = pd.read_excel(file)


#%%

plt.style.use(['seaborn-paper','seaborn-whitegrid'])
plt.rc("font", family="serif")
#plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
width  = 6
height = width / 1.618

   
#%% Plot all at once 3x3
#   Plot 1/4

# Plot a single orbit
shift = 56-500
if case == 'max_ebatt':
    start = 910 + shift # 900+7+2 # int(120*60/15)
    length = 54 # 59 #55 # int(9*60/15)
elif case == 'max_te':
    start = 910+29 + shift # 900+7+2 # int(120*60/15)
    length = 54+2 # 59 #55 # int(9*60/15)    
else:
    print('No case defined')

end = start + length
print(d.time[start:end].reset_index(drop=True)[0]/60)

xpts = np.arange(1,len(d.h[start:end])+1)*8/60 # This is now time in min

plt.figure(figsize=(6,6))
#plt.title('Complete Orbit (2 hrs after dawn)')
y = d.x[start:end].reset_index(drop=True)/1000 # Flipped to match a map
x = d.y[start:end].reset_index(drop=True)/1000
plt.plot(x, y, color = 'C0', linestyle = '-', linewidth = 1, alpha=0.5)
plt.scatter(x, y, c=xpts, cmap=plt.cm.viridis, s=50)
plt.xlim([-3,3])
plt.ylim([-3,3])
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)

ax.tick_params(axis = 'both', labelsize=13)

circle1 = plt.Circle((0, 0), 3, color='k', fill=False, alpha=0.5, linestyle='--', linewidth=2)
ax.add_artist(circle1)

azimuth = (d.azimuth[start:end].mean()) * np.pi / 180
#plt.plot([0,1*np.cos(azimuth)], [0,1*np.sin(azimuth)], color = 'C1')
plt.plot([0,1*np.sin(azimuth)], [0,1*np.cos(azimuth)], color = 'C4') # Switched directions
plt.scatter(0,0, s = 100, color = 'C4')

plt.ylabel('N (km)')
plt.xlabel('E (km)')

#for i in range(7,len(xpts),8):
#    if i == 22:
#        plt.text(x[i]-0.15, 
#                 y[i]+0.15, 
#                 '{:.1f}'.format(xpts[i]),
#                 size = 10)
#    else:        
#        plt.text(x[i]+0.05, 
#                 y[i]+0.1, 
#                 '{:.1f}'.format(xpts[i]),
#                 size = 10)

for i in [14, 29, 44]:
    plt.text(x[i]+0.05, 
             y[i]+0.1, 
             '{:.0f}'.format(xpts[i]),
             size = 13)

for i in [6, 36]:
       plt.text((x[i]+x[i+1])/2+0.05, 
             (y[i]+y[i+1])/2+0.1, 
             '{:.0f}'.format(xpts[i]),
             size = 13) 

for i in [21]:#, 51]:
       plt.text((x[i]+x[i+1])/2+3*0.05, 
             (y[i]+y[i+1])/2+0.1, 
             '{:.0f}'.format(xpts[i]),
             size = 13) 

for i in [51]:
       plt.text((x[i]+x[i+1])/2+2*0.05, 
             (y[i]+y[i+1])/2-0.1, 
             '{:.0f}'.format(xpts[i]),
             size = 13)        

plt.text(3.5,0,'0', color='w')
plt.text(-4.2,0,'0', color='w')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(12)

plt.tight_layout()

if save_plots == True:
    plt.savefig('winter_orbit_'+str(name)+'_1'+'.pdf', bbox_inches='tight') ## FINISH HERE

#%% Orbit energy anaysis

orbit_power_needed = integrate.simps(d.p_n[start:end], d.time[start:end]) / 10**6
orbit_solar_power = integrate.simps(d.p_solar[start:end], d.time[start:end]) / 10**6
orbit_total_energy = integrate.simps(d.te[start:end], d.time[start:end]) / 10**3
orbit_battery_energy = integrate.simps(d.e_batt[start:end], d.time[start:end]) / 10**3
orbit_solar_power / orbit_power_needed
print(orbit_power_needed) # MJ
print(orbit_solar_power)
print(orbit_total_energy) # kJ
print(orbit_battery_energy)

#%% Plot 2/4

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

plt.figure(figsize=(6,5))

yscale = 0.25

xmax = 7.5 # 8

plt.subplot(3,1,1)
y = d.h[start:end]*3.28084/1000
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Height')
plt.ylabel('$kft$')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,2)
y = np.degrees(d.alpha[start:end])
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Angle of Attack')
plt.ylabel('Deg')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,3)
y = np.degrees(d.phi[start:end])
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Bank Angle')
plt.ylabel('Deg')
plt.xlabel('Time (min)')
plt.xlim([0,xmax])

plt.tight_layout()

if save_plots == True:
    plt.savefig('winter_orbit_'+str(name)+'_2'+'.pdf', bbox_inches='tight')

#%% Plot 3/4

plt.figure(figsize=(6,5))

plt.subplot(3,1,1)
y = d.p_solar[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Solar Power Received')
plt.ylabel('$W$')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,2)
y = d.p_n[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Power Needed')
plt.ylabel('$W$')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,3)
y = d.p_bat[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Power to Battery')
plt.ylabel('$W$')
plt.xlabel('Time (min)')
plt.xlim([0,xmax])

plt.tight_layout()
if save_plots == True:
    plt.savefig('winter_orbit_'+str(name)+'_3'+'.pdf', bbox_inches='tight')

#%% Plot 4/4
plt.figure(figsize=(6,5))


plt.subplot(3,1,1)
y = d.v[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Velocity')
plt.ylabel('$m/s$')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,2)
y = d.tp[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
ax.set_xticklabels([])
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Thrust')
plt.ylabel('$N$')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0,xmax])

plt.subplot(3,1,3)
y = d.d[start:end]
plt.scatter(xpts, y, c=xpts, cmap=plt.cm.viridis)
ax = plt.gca()
ax.grid(linestyle='-', linewidth=1)
plt.xlim([0,xpts[-1]+15/60])
ymin = y.min() - yscale*(y.max()-y.min())
ymax = y.max() + yscale*(y.max()-y.min())
plt.ylim([ymin, ymax])
plt.title('Drag')
plt.ylabel('$N$')
plt.xlabel('Time (min)')
plt.xlim([0,xmax])

plt.tight_layout()
if save_plots == True:
    plt.savefig('winter_orbit_'+str(name)+'_4'+'.pdf', bbox_inches='tight')

