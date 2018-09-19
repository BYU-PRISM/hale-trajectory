#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:52:50 2018

@author: nathanielgates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#winter_folder = 'hale_2018_07_01_00_07_38 - Winter Timestep 8 sec Horizon 15 min CL 1.5'
#spring_folder = 'hale_2018_07_01_00_18_43 - Spring Timestep 8 sec Horizon 15 min CL 1.5'
#summer_folder = 'hale_2018_07_01_00_19_26 - Summer Timestep 8 sec Horizon 15 min CL 1.5'
#fall_folder = 'hale_2018_07_01_00_20_06 - Fall Timestep 8 sec Horizon 15 min CL 1.5'

# For testing
summer_folder = 'summer - opt_results_2018_07_01_16_43_30.xlsx'
winter_folder = 'winter - opt_results_2018_07_01_16_19_38.xlsx'

# For August
summer_folder = r'C:\Users\PRISM Lab\Documents\GitHubCode\hale-optimization\Trajectory\Data\hale_2018_08_27_12_16_19 - Summer Battery 136\Intermediates\iter_10660.xlsx'
winter_folder = r'C:\Users\PRISM Lab\Documents\GitHubCode\hale-optimization\Trajectory\Data\hale_2018_08_27_12_16_00 - Winter Battery 136\Intermediates\iter_10660.xlsx'

s = pd.read_excel(summer_folder)
w = pd.read_excel(winter_folder)


plt.style.use(['seaborn-paper','seaborn-whitegrid'])
plt.rc("font", family="serif")
#plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
width  = 6
height = width / 1.618

def size_axis(font_size=12):
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

fontsize = 12

save_plots = True

#%% Plot all on one plot

ymax = np.max([s.p_solar, s.p_n, (s.p_solar - s.p_n), 
               w.p_solar, w.p_n, (w.p_solar - w.p_n)])/1000
    
ymin = np.min([s.p_solar, s.p_n, (s.p_solar - s.p_n), 
               w.p_solar, w.p_n, (w.p_solar - w.p_n)])/1000

yscale = ymax - ymin

for i in range(2):
    plt.figure(figsize=(7,5))

    if i == 0:
        d = s
        plt.title('Summer')
    else:
        d = w
        plt.title('Winter') 
    
    alpha = 0.8
    linestyle = [':','--','-']
    
#    alpha = 1
    linestyle = ['-','-','-']
    
    
    d['soc'] = d.e_batt/(d.e_batt[0]/0.2)    
    full = np.where(d.soc >= 1)[0][0]
    sunset = len(d) - np.where(d.flux.iloc[::-1] == 0)[0][-1] - 1
    loiter = np.where(d.h[sunset:] == d.h.min())[0][0] + sunset
    
    idx = [full, sunset, loiter]
    for i in range(len(idx)):
#        plt.axvline((d.time/3600)[idx[i]], c = 'k', linestyle = '--', alpha = 0.5)
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (-5, 21.4), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.17, 21.4, str(i+1))

    colors = ['C0','C1','C2']
#    colors = ['gray','lightgray','k']
    
    plt.plot(d.time/3600, d.p_solar/1000, c = colors[0], label = 'Solar Power', alpha = alpha, linestyle=linestyle[0]) #s.te/3.6)
    plt.plot(d.time/3600, d.p_n/1000, c = colors[1], label = 'Power Needed', alpha = alpha, linestyle = linestyle[1])
#    plt.plot(d.time/3600, (d.p_solar - d.p_n)/1000, c = colors[2], label = 'Net Power', alpha = alpha, linestyle = linestyle[2])
    plt.plot(d.time/3600, d.p_bat/1000, c = colors[2], label = 'P_batt', alpha = alpha, linestyle = linestyle[2])

    plt.xlim(0,24)
    plt.xticks(np.arange(0,27,3))
    plt.ylim(-5, ymax + 0.05*yscale)
    plt.xlabel('Time (hr)')
    plt.ylabel('Power (kW)')
    
    plt.tight_layout()
    plt.legend(frameon=True,prop={'size': fontsize})
    
    size_axis(fontsize)


#%% Plot on 2x2 subplot

#ymax = 16
y_text = 15
y_text_1 = 19.3
y_text_2 = 24.05
line_height_1 = 19
line_height_2 = 24
yticks = np.arange(0,20,5)

for i in range(2):

    plt.figure(figsize=(7,5))
#    plt.figure(figsize=(9,7))    
        
    if i == 0:
        ymax_save = ymax
        d = s
#        plt.suptitle('Summer')
        name = 'Summer'
    else:
        d = w
#        plt.suptitle('Winter') 
        name = 'Winter'
        
        # For Winter alone
        scale = 16/21.025369999999999
        ymax = 16
        y_text_1 = 19.3*scale
        y_text_2 = 24.05
        line_height_1 = 19*scale
        line_height_2 = 24
        #yticks = np.arange(0,25,5)
    
    d['soc'] = d.e_batt/(d.e_batt[0]/0.2)
    
    full = np.where(d.soc >= 1)[0][0]
    sunset = len(d) - np.where(d.flux.iloc[::-1] == 0)[0][-1] - 1
    loiter = np.where(d.h[sunset:] == d.h.min())[0][0] + sunset
    
    idx = [full, sunset, loiter]
    
    alpha = 0.9
    
    plt.subplot(2,2,1)
    for i in range(len(idx)):
#        plt.axvline((d.time/3600)[idx[i]], c = 'k', linestyle = ':')
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, line_height_1), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, y_text_1, str(i+1))
    plt.plot(d.time/3600, d.p_solar/1000, c = 'C0', label = 'Solar Power', alpha = alpha) #s.te/3.6)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    #plt.xlabel('Time (hr)')
    plt.ylabel('Solar Power (kW)')
    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
    plt.yticks(yticks)
    size_axis(fontsize)
    
    plt.subplot(2,2,2)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, line_height_1), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, y_text_1, str(i+1))
    plt.plot(d.time/3600, d.p_n/1000, c = 'C1', label = 'Power Needed', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
    plt.yticks(yticks)
    #plt.xlabel('Time (hr)')
    plt.ylabel('Power Needed (kW)')
    size_axis(fontsize)
    
    plt.subplot(2,2,3)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, line_height_1), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, y_text_1, str(i+1))
#    plt.plot(d.time/3600, (d.p_solar - d.p_n)/1000, c = 'C2', label = 'Net Power', alpha = alpha)
    plt.plot(d.time/3600, d.p_bat/1000, c = 'C2', label = 'Net Power', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
    plt.yticks(yticks)
    plt.xlabel('Time (hr)')
#    plt.ylabel('Net Power (kW)')
    plt.ylabel('Power to Battery (kW)')
    size_axis(fontsize)
    
    plt.subplot(2,2,4)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, line_height_2), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, y_text_2, str(i+1))
    plt.plot(d.time/3600, d.h/1000, c = 'C3', label = 'Height', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    hmin = np.min([s.h, w.h])/1000
    hmax = np.max([s.h, w.h])/1000
    hscale = hmax - hmin
    plt.ylim(hmin - 0.05*hscale, hmax + 0.05*hscale)
    plt.xlabel('Time (hr)')
    plt.ylabel('Height (km)')
    size_axis(fontsize)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plots == True:
        plt.savefig('day_detail_'+str(name)+'.pdf', bbox_inches='tight')

ymax = ymax_save
    
#%% Plot additioanl details (on 2x2 subplot)

for i in range(2):

    plt.figure(figsize=(7,5))
#    plt.figure(figsize=(9,7))    
        
    if i == 0:
        d = s
        plt.suptitle('Summer')
    else:
        d = w
        plt.suptitle('Winter') 
    
    d['soc'] = d.e_batt/(d.e_batt[0]/0.2)
    
    full = np.where(d.soc >= 1)[0][0]
    sunset = len(d) - np.where(d.flux.iloc[::-1] == 0)[0][-1] - 1
    loiter = np.where(d.h[sunset:] == d.h.min())[0][0] + sunset
    
    idx = [full, sunset, loiter]
    
    alpha = 0.9
    
    plt.subplot(2,2,1)
    for i in range(len(idx)):
#        plt.axvline((d.time/3600)[idx[i]], c = 'k', linestyle = ':')
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, 19.6), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, 20, str(i+1))
    plt.plot(d.time/3600, d.v, c = 'C0', label = 'Velocity', alpha = alpha) #s.te/3.6)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    #plt.xlabel('Time (hr)')
    plt.ylabel('Velocity (m/s)')
    plt.ylim(25,55)
#    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
#    plt.yticks(np.arange(0,25,5))
    size_axis(fontsize)
    
    plt.subplot(2,2,2)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, 19.6), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, 20, str(i+1))
    plt.plot(d.time/3600, np.degrees(d.alpha), c = 'C1', label = 'Angle of Attack', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    plt.ylim(0,15)
#    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
#    plt.yticks(np.arange(0,25,5))
    #plt.xlabel('Time (hr)')
    plt.ylabel('Angle of Attack (deg)')
    size_axis(fontsize)
    
    plt.subplot(2,2,3)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, 19.6), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, 20, str(i+1))
#    plt.plot(d.time/3600, (d.p_solar - d.p_n)/1000, c = 'C2', label = 'Net Power', alpha = alpha)
    plt.plot(d.time/3600, d.p_bat/1000, c = 'C2', label = 'Net Power', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    plt.ylim(ymin - 0.02*yscale, ymax + 0.02*yscale)
    plt.yticks(np.arange(0,25,5))
    plt.xlabel('Time (hr)')
#    plt.ylabel('Net Power (kW)')
    plt.ylabel('Power to Battery (kW)')
    size_axis(fontsize)
    
    plt.subplot(2,2,4)
    for i in range(len(idx)):
        plt.plot(((d.time/3600)[idx[i]], (d.time/3600)[idx[i]]), 
                 (ymin - 0.02*yscale, 24), 'k:')
        plt.text((d.time/3600)[idx[i]]-0.35, 24.1, str(i+1))
    plt.plot(d.time/3600, d.h/1000, c = 'C3', label = 'Height', alpha = alpha)
    plt.xlim(0,24)
    plt.xticks(np.arange(0,30,6))
    hmin = np.min([s.h, w.h])/1000
    hmax = np.max([s.h, w.h])/1000
    hscale = hmax - hmin
    plt.ylim(hmin - 0.05*hscale, hmax + 0.05*hscale)
    plt.xlabel('Time (hr)')
    plt.ylabel('Height (km)')
    size_axis(fontsize)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])    
