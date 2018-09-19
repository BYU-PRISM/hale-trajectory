# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../Sandbox/FFT/")
from plotting import plot3DPath, plotSolar, plotTotalEnergy, plot2DPath_Labeled, plot3DPath_NorthSouth, plot2DPath_Radius, plot2DPath_NoLabel
from findOrbits import find_orbits
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from dashboard import fix_units
import yaml
from itertools import cycle


plt.style.use(['seaborn-paper','seaborn-whitegrid'])
plt.rc("font", family="serif")
#plt.rc('text', usetex=False)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
width  = 4
height = width / 1.618

plt.close('all')

#%%  Data
#winter_folder = 'hale_2018_02_28_11_02_58 - E216 Winter Drag'
#summer_folder = 'hale_2018_02_28_11_03_13 - E216 Summer Drag'
#spring_folder = 'hale_2018_02_28_11_03_05 - E216 Spring Drag'
#fall_folder = 'hale_2018_02_28_11_03_21 - E216 Fall Drag'
#winter_folder = 'hale_2018_04_23_07_54_18 - E216 CL 1.1 Winter'
#summer_folder = 'hale_2018_04_23_08_01_21 - E216 CL 1.1 Summer'
#spring_folder = 'hale_2018_04_23_08_01_10 - E216 CL 1.1 Spring'
#fall_folder = 'hale_2018_04_23_08_01_28 - E216 CL 1.1 Fall'

winter_folder = 'hale_2018_08_27_12_16_00 - Winter Battery 136'
spring_folder = 'hale_2018_08_27_12_16_10 - Spring Battery 136'
summer_folder = 'hale_2018_08_27_12_16_19 - Summer Battery 136'
fall_folder = 'hale_2018_08_27_12_16_26 - Fall Battery 136'

data={}
data['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/'+winter_folder+'/opt*.xlsx')[-1]))
data['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/'+summer_folder+'/opt*.xlsx')[-1]))
data['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/'+spring_folder+'/opt*.xlsx')[-1]))
data['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/'+fall_folder+'/opt*.xlsx')[-1]))

ss={}
ss['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/'+winter_folder+'/ss*.xlsx')[-1]))
ss['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/'+summer_folder+'/ss*.xlsx')[-1]))
ss['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/'+spring_folder+'/ss*.xlsx')[-1]))
ss['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/'+fall_folder+'/ss*.xlsx')[-1]))

sm={}
#sm['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_03_02_14_21_44 - E216 SM Winter H Opt/sm*.xlsx')[-1]))
#sm['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_03_01_11_30_09 - E216 SM Summer H Opt/sm*.xlsx')[-1]))
#sm['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_03_01_11_33_36 - E216 SM Spring H Opt/sm*.xlsx')[-1]))
#sm['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_03_01_11_36_08 - E216 SM Fall H Opt/sm*.xlsx')[-1]))
sm['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_08_27_15_26_34 - SM Winter Battery 136/sm*.xlsx')[-1]))
sm['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_08_27_15_27_04 - SM Summer Battery 136/sm*.xlsx')[-1]))
sm['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_08_27_15_26_48 - SM Spring Battery 136/sm*.xlsx')[-1]))
sm['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_08_27_15_27_15 - SM Fall Battery 136/sm*.xlsx')[-1]))

rad = {}
rad['6000'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_02_26_12_44_05 - E216 Winter 6000/Intermediates/*.xlsx')[-1]))
rad['12000'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_02_25_23_23_08 - E216 Winter 12000/Intermediates/*.xlsx')[-1]))
rad['1500'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_02_25_23_23_27 - E216 Winter 1500/Intermediates/*.xlsx')[-1]))

#SMS = {}
#SMS['D1'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_13_26_54 - SM Summer D 1/ss*.xlsx')[-1]))
#SMS['D2'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_04_53 - SM Summer D 2/ss*.xlsx')[-1]))
#SMS['D3'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_15_00 - SM Summer D 3/ss*.xlsx')[-1]))

#SMF = {}
#SMS['D1'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_29_27 - SM Fall D 1/ss*.xlsx')[-1]))
#SMS['D2'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_29_10 - SM Fall D 2/ss*.xlsx')[-1]))
#SMS['D3'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_28_48 - SM Fall D 3/ss*.xlsx')[-1]))

yaml_file = glob.glob('../Data/'+winter_folder+'/config*')[0]
with open(yaml_file, 'r') as ifile:
        config = yaml.load(ifile)

#%% Solar Plots
fig = plt.figure()
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
for season in data:
    df = data[season]
    df = df.loc[df['flux']>0.01]
    
    plt.plot(df.t/3600,df.flux,next(linecycler),label=season)
    
plt.ylim([0,1400])
plt.title('Total Solar Flux Available')
plt.xlabel('Time (Hr)')
plt.ylabel('Available Flux (W/m$^2$)')
plt.legend()
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('available_flux.pdf', bbox_inches='tight')

# Available solar flux Winter
fig = plt.figure()
df = data['Winter']
df = df.loc[df['flux']>0.01]
plt.plot(df.t_hr,df.flux,c='k')
plt.ylim([0,1400])
#plt.title('Total Solar Flux Available')
plt.xlabel('Time (Hr)')
plt.ylabel('Available Flux (W/m$^2$)')
#plt.legend()
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('available_winter_flux.pdf', bbox_inches='tight')

# Solar power recieved ss winter
fig = plt.figure()
df = ss['Winter']
#df = df.loc[df['flux']>0.01]
plt.plot(df.t_hr,df.p_solar,c='k')
#plt.ylim([0,900])
#plt.title('Total Solar Flux Available')
plt.xlabel('Time (Hr)')
plt.ylabel('Solar Power Recieved (W)')
#plt.legend()
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('winter_ss_flux.pdf', bbox_inches='tight')

fig = plt.figure()
for season in data:
    df = data[season]
    df = df.loc[df['flux']>0.01]

    
    # Azimuth and Zenith
    fig = plt.figure()
    plt.plot(df.time_hr,df.azimuth,c='k',label='Sun Azimuth')
    plt.plot(df.time_hr,df.zenith,'--',color="0.5",label='Sun Zenith')
    plt.legend(loc='best')
    plt.title('Solar Azimuth and Zenith (' + season + ')')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Angle (Degrees)')
    plt.xlim([0,15])
    plt.ylim([0, 300])
    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('azimuth_zenith_'+season+'.pdf', bbox_inches='tight')
    
    
#%% Total Energy
    
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
for season in data:
    df = data[season]
    df_ss = ss[season]
    df_sm = sm[season]
    
    # Total Energy and Battery Energy
    fig = plt.figure()
#    plt.plot(df.time_hr,df.te_kwh,':',color='k',label='Optimized Total Energy')
#    plt.plot(df.time_hr,df.e_batt_kwh,'-',color="0.5",label='Optimized Battery Energy')
#    plt.plot(df_ss.time_hr,df_ss.te_kwh,'-',color="k",label='Circular Total Energy')
#    plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,'-.',color="0.5",label='Circular Battery Energy')
    plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy')
    plt.plot(df.time_hr,df.e_batt_kwh,'--',label='Optimized Battery Energy')
#    plt.plot(df_ss.time_hr,df_ss.te_kwh,'-',label='SS Total Energy')
    plt.plot(df_ss.time_hr,df_sm.te_kwh,'-.',label='SM Total Energy')
    plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,':',label='SS Battery Energy')
    plt.legend(loc='best')
    plt.title('Energy Storage (' + season + ')')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Energy Stored (kWh)')
    plt.xlim([0,24])
    plt.ylim([0,70])
    plt.xticks([0,6,12,18,24])
    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('total_energy_'+season+'.pdf', bbox_inches='tight')
    
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
    
##%% State Machine total energy
#fig = plt.figure()
#df = data['Summer']
#df_ss = ss['Summer']
#plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy')
#plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,'-.',label='SS Battery Energy')
#df = SMS['D1']
#plt.plot(df.time_hr,df.te_kwh,label='State Machine 1 Total Energy')
#df = SMS['D2']
#plt.plot(df.time_hr,df.te_kwh,label='State Machine 2 Total Energy')
#df = SMS['D3']
#plt.plot(df.time_hr,df.te_kwh,label='State Machine 3 Total Energy')
#plt.legend(loc='best')
#plt.title('State Machine Energy Storage Comparison (Summer)')
#plt.xlabel('Time (Hr)')
#plt.ylabel('Energy Stored (kWh)')
#plt.xlim([0,24])
#plt.xticks([0,6,12,18,24])
#plt.ylim([0,70])
#plt.tight_layout()
#fig.set_size_inches(width, height)
#plt.savefig('total_energy_state_machine.pdf', bbox_inches='tight')
    
    
#%% Total Energy Table
df_table = pd.DataFrame(columns=['season','trajectory','max','final','change'])
for season in data:
    df_ss = ss[season]
    row = pd.Series([season,'Steady State',df_ss.te_kwh.max(),df_ss.te_kwh.tail(1).iloc[0],''],['season','trajectory','max','final','change'])
    df_table =df_table.append([row],ignore_index=True)
    df_sm = sm[season]
    change_sm = round(df_sm.te_kwh.tail(1).iloc[0]-df_ss.te_kwh.tail(1).iloc[0],2)
    row = pd.Series([season,'State Machine',df_sm.te_kwh.max(),df_sm.te_kwh.tail(1).iloc[0],change_sm],['season','trajectory','max','final','change'])
    df_table = df_table.append([row],ignore_index=True)
    df = data[season]
    change = round(df.te_kwh.tail(1).iloc[0]-df_ss.te_kwh.tail(1).iloc[0],2)
    row = pd.Series([season,'Optimized',df.te_kwh.max(),df.te_kwh.tail(1).iloc[0],change],['season','trajectory','max','final','change'])
    df_table = df_table.append([row],ignore_index=True)
    
df_table.columns = ['Season','Trajectory', 'Max Total Energy (kWh)','Final Total Energy (kWh)','Improvement (kWh)']
df_table = df_table.round(2)
print('Total Energy Table')
print(df_table.to_latex(index=False))

#%% Total Energy Bar Chart
fig, ax = plt.subplots()
xlabels = []
for season in data:
    xlabels.append(season)
index = np.arange(len(xlabels))
barwidth = 0.25
    
#plt.bar(index,df_table.loc[(df_table['Trajectory']=='Steady State')]['Max Total Energy (kWh)'],
#        width = barwidth,
#        alpha = 0.8,
#        label='SS Max Total Energy')
#plt.bar(index+barwidth,df_table.loc[(df_table['Trajectory']=='State Machine')]['Max Total Energy (kWh)'],
#        width = barwidth,
#        alpha = 0.8,
#        label='SM Max Total Energy')
#plt.bar(index+barwidth*2,df_table.loc[(df_table['Trajectory']=='Optimized')]['Max Total Energy (kWh)'],
#        width = barwidth,
#        alpha = 0.8,
#        label='Opt Max Total Energy')
plt.bar(index,df_table.loc[(df_table['Trajectory']=='Steady State')]['Final Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='SS Final Total Energy')
plt.bar(index+barwidth,df_table.loc[(df_table['Trajectory']=='State Machine')]['Final Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='SM Final Total Energy')
plt.bar(index+barwidth*2,df_table.loc[(df_table['Trajectory']=='Optimized')]['Final Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='Opt Final Total Energy')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='right', bbox_to_anchor=(1.4,0.5))
plt.xlabel('Season')
plt.ylabel('Total Energy (kWh)')
plt.xticks(index+barwidth/2,xlabels)
plt.tight_layout()
fig.set_size_inches(width*1.5, height)
plt.savefig('total_energy_bar.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
#%% Elevation Profiles
fig = plt.figure()
for season in data:
    df = data[season]
    plt.plot(df.time_hr,df.h/1000,label=season)
plt.legend()
plt.title('Optimal Trajectory Altitude')
plt.xlabel('Time (hr)')
plt.ylabel('Altitude (km)')
plt.xlim([0,24])
plt.ylim([18,25])
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('altitude.pdf', bbox_inches='tight')
    
    
#%%     
# Variables vs timestep
horizon = 75
vph = 3750
eph = 3300
vps = vph/horizon
eps = eph/horizon
neqs = []
nvars = []
trange = range(1,3600)
for timestep in trange:
    steps = 3600*24/timestep
    neqs.append(steps*eps)
    nvars.append(steps*vps)
fig = plt.figure()
plt.plot(trange,neqs,label='# Equations',c='k')
plt.plot(trange,nvars,'--',label='# Variables',c='k')
plt.yscale('log')
plt.xscale('log')
plt.title('Problem Size vs Timestep Size')
plt.xlabel('Timestep size (s)')
plt.legend()
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('scaling.pdf', bbox_inches='tight')

#%%
# Mu Effect
df = data['Winter']
fig = plt.figure()
for mu in reversed(np.arange(0,1.25,0.25)):
    plt.plot(df.time_hr,df.flux*mu,label='$\mu_{solar}$ = '+str(mu))
plt.legend()
plt.title('Effect of Obliquity factor on Winter Solar Flux')
plt.xlabel('Time (hr)')
plt.ylabel('Solar Flux (W/m$^2$)')
fig.set_size_inches(width, height)
plt.savefig('mu_effect.pdf', bbox_inches='tight')

#%%# Power Required

#fig = plt.figure()
#for season in data:
#    df = data[season]
#    plt.plot(df.time_hr,df.p_n,label=season)
#plt.legend()
#plt.title('Power Required (W)')
#plt.xlabel('Time (hr)')
#plt.ylabel('Power Required (W)')
#plt.tight_layout()
#fig.set_size_inches(width, height)
#plt.savefig('power_required.pdf', bbox_inches='tight')

#plt.style.use(['seaborn-paper','seaborn-whitegrid'])

#fig, axarr = plt.subplots(2, 2)
#season = 'Winter'
#df = data[season]
#axarr[0, 0].plot(df.time_hr,df.p_n)
#axarr[0, 0].set_title(season)
#axarr[0, 0].set_ylabel('Power Required (W)')
#season = 'Spring'
#df = data[season]
#axarr[0, 1].plot(df.time_hr,df.p_n)
#axarr[0, 1].set_title(season)
#season = 'Summer'
#df = data[season]
#axarr[1, 0].plot(df.time_hr,df.p_n)
#axarr[1, 0].set_title(season)
#axarr[1, 0].set_xlabel('Time (hr)')
#axarr[1, 0].set_ylabel('Power Required (W)')
#season = 'Fall'
#df = data[season]
#axarr[1, 1].plot(df.time_hr,df.p_n)
#axarr[1, 1].set_title(season)
#axarr[1, 1].set_xlabel('Time (hr)')
## Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
#plt.tight_layout()
#fig.set_size_inches(width, height)
#plt.savefig('power_required.pdf', bbox_inches='tight')

# Power in - Power Out
plt.style.use(['seaborn-paper','seaborn-whitegrid'])
fig, axarr = plt.subplots(2, 2)
season = 'Winter'
df = data[season]
axarr[0,0].axvline(x=6.6,linestyle=':',color='0.1')
axarr[0,0].text(5.3,15500,'1',size='xx-small')
axarr[0,0].axvline(x=9.7,linestyle=':',color='0.1')
axarr[0,0].text(8.4,15500,'2',size='xx-small')
axarr[0,0].axvline(x=10.9,linestyle=':',color='0.1')
axarr[0,0].text(11.5,15500,'3',size='xx-small')
axarr[0, 0].plot(df.time_hr,df.pinout)
axarr[0, 0].set_title(season)
axarr[0, 0].set_ylabel('Net Power (W)')
axarr[0, 0].set_xlim([0,24])
axarr[0, 0].set_ylim([-5000,18000])
season = 'Spring'
df = data[season]
axarr[0,1].axvline(x=5.1,linestyle=':',color='0.1')
axarr[0,1].text(3.9,15500,'1',size='xx-small')
axarr[0,1].axvline(x=12.0,linestyle=':',color='0.1')
axarr[0,1].text(10.8,15500,'2',size='xx-small')
axarr[0,1].axvline(x=13.4,linestyle=':',color='0.1')
axarr[0,1].text(14.1,15500,'3',size='xx-small')
axarr[0, 1].plot(df.time_hr,df.pinout)
axarr[0, 1].set_title(season)
axarr[0, 1].set_xlim([0,24])
axarr[0, 1].set_ylim([-5000,18000])
season = 'Summer'
df = data[season]
axarr[1,0].axvline(x=5.2,linestyle=':',color='0.1')
axarr[1,0].text(3.8,15500,'1',size='xx-small')
axarr[1,0].axvline(x=14.4,linestyle=':',color='0.1')
axarr[1,0].text(13,15500,'2',size='xx-small')
axarr[1,0].axvline(x=15.7,linestyle=':',color='0.1')
axarr[1,0].text(16.3,15500,'3',size='xx-small')
axarr[1, 0].plot(df.time_hr,df.pinout)
axarr[1, 0].set_title(season)
axarr[1, 0].set_xlabel('Time (hr)')
axarr[1, 0].set_ylabel('Net Power (W)')
axarr[1, 0].set_xlim([0,24])
axarr[1, 0].set_ylim([-5000,18000])
season = 'Fall'
df = data[season]
axarr[1,1].axvline(x=5.1,linestyle=':',color='0.1')
axarr[1,1].text(3.9,15500,'1',size='xx-small')
axarr[1,1].axvline(x=12.0,linestyle=':',color='0.1')
axarr[1,1].text(10.8,15500,'2',size='xx-small')
axarr[1,1].axvline(x=13.4,linestyle=':',color='0.1')
axarr[1,1].text(14.1,15500,'3',size='xx-small')
axarr[1, 1].plot(df.time_hr,df.pinout)
axarr[1, 1].set_title(season)
axarr[1, 1].set_xlabel('Time (hr)')
axarr[1, 1].set_xlim([0,24])
axarr[1, 1].set_ylim([-5000,18000])
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('power_inout.pdf', bbox_inches='tight')

#plt.style.use(['seaborn-notebook','seaborn-whitegrid'])

#%%
# Rate of climb
fig = plt.figure()
for season in data:
    df = data[season]
    time_step = df.time[1]-df.time[0]
    plt.plot(df.time_hr,round(df.h.diff()/time_step,2),label=season)
plt.legend()
plt.title('Rate of Climb')
plt.xlabel('Time (hr)')
plt.ylabel('Rate of Climb (m/s)')
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('rate_of_climb.pdf', bbox_inches='tight')

#%% Panel Efficiency
fig = plt.figure()
eta = config['solar']['panel_efficiency_function']['eta'] # 0.12
beta = config['solar']['panel_efficiency_function']['beta'] # 0.0021888986107182 # 0.002720315 (old)
Tref = config['solar']['panel_efficiency_function']['Tref'] # 298.15
gamma_s = config['solar']['panel_efficiency_function']['gamma_s'] # 0.413220518404272 # 0.513153469 (old)
T_noct = config['solar']['panel_efficiency_function']['T_noct'] # 20.0310337470507 # 20.0457889 (old)
G_noct = config['solar']['panel_efficiency_function']['G_noct'] # 0.519455027587048 # 0.527822252 (old)
T_11 = 216.66 # Standard air temp at 11 km (K)
pe_list = []
for G_sol in range(0,1500):
    panel_efficiency = eta*(1-beta*(T_11-Tref+(T_noct-20)*G_sol/G_noct)+gamma_s*np.log10(G_sol+0.01))
    pe_list.append(panel_efficiency)
plt.plot(range(0,1500),pe_list,'k')
plt.title('Solar Panel Efficiency')
plt.xlabel('Flux (W/m^2)')
plt.ylabel('Efficiency')
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('solar_eff.pdf', bbox_inches='tight')

#%% Single Orbits
#plt.style.use(['seaborn-notebook','seaborn-whitegrid'])
for season in data:
    df = data[season]
    x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(df)
    df = df.loc[(df['time']>=t_orbits[-3]*0.999) & (df['time']<=t_orbits[-2]*1.001)]
#    plot2DPath_Labeled(df,season,config)
    plot2DPath_NoLabel(df,season,config)
    # Solar Flux
    fig = plt.figure()
    plt.plot(df['time_hr'],df['p_solar'])
    plt.xlabel('Time (hr)')
    plt.ylabel('Solar Flux Recieved (W)')
    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('single_solar_'+season+'.pdf', bbox_inches='tight')
    # pinout
    fig = plt.figure()
    plt.plot(df['time_hr'],df['pinout'])
    plt.xlabel('Time (hr)')
    plt.ylabel('Net Power (W)')
    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('net_power_'+season+'.pdf', bbox_inches='tight')
    
##%% Time series data
#season = 'Winter'
#df = data[season]
#x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(df)
#df = df.loc[(df['time']>=t_orbits[-2]) & (df['time']<=t_orbits[-1])]
#df['time0'] = df['time']-df['time'].values[0]
#plt.figure()
#plt.plot(df['time0'],df['p_n'])
#plt.plot(df['time0'],df['p_bat'])
#plt.plot(df['time0'],df['p_solar'])
#plt.xlabel('Time (s)')
#plt.ylabel('Power (W)')
#plt.legend()
#plt.figure()
#plt.plot(df['time0'],df['alpha_deg'])
#plt.plot(df['time0'],df['phi_deg'])
#plt.xlabel('Time (s)')
#plt.ylabel('Angle (deg)')
#plt.legend()
#plt.figure()
#plt.plot(df['time0'],df['d'])
#plt.plot(df['time0'],df['tp'])
#plt.xlabel('Time (s)')
#plt.ylabel('Angle (deg)')
#plt.legend()

#%% Full Orbits
#plt.style.use(['seaborn-notebook','seaborn-whitegrid'])
for season in data:
    ax, fig = plot3DPath_NorthSouth(data[season],tight=True)
#    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('3d_path_'+season+'.pdf', bbox_inches='tight')
    
## State Machine
ax, fig = plot3DPath_NorthSouth(sm['Summer'],tight=True)
fig.set_size_inches(width*1.2, height*1.2)
plt.savefig('3d_path_StateMachine.pdf', bbox_inches='tight')

#plt.style.use(['seaborn-notebook','seaborn-whitegrid'])

#%% Steady State Orbit
df = ss['Winter']
ax, fig = plot3DPath_NorthSouth(df,tight=False)
fig.set_size_inches(width, height)
plt.savefig('ss_path_winter.pdf', bbox_inches='tight')

#%% Radius Plots
#r6 = pd.read_excel('rad6000.xlsx')
r6 = rad['6000'].iloc[264:329,:]
r15 = rad['1500'].iloc[279:301,:]
#r15 = pd.read_excel('rad1500.xlsx')
#r12 = pd.read_excel('rad12000.xlsx')
plot2DPath_Radius(r6,6000)
plot2DPath_Radius(r15,1500)
#plot2DPath_Radius(r12,12000)
#for radius in rad:
#    df = rad[radius]
#    x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(df)
#    df = df.loc[(df['time']>=t_orbits[-4]) & (df['time']<=t_orbits[-3])]
#    plot2DPath_Radius(df,radius)

# Radius Energy
season = 'Winter'
df = data[season]
r6 = rad['6000']
r15 = rad['1500']
fig = plt.figure()
plt.plot(r15.time_hr,r15.te_kwh,label='Optimized Total Energy 1.5 km')
plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy 3 km')
plt.plot(r6.time_hr,r6.te_kwh,label='Optimized Total Energy 6 km')

plt.legend(loc='best')
plt.title('Energy Storage (' + season + ')')
plt.xlabel('Time (Hr)')
plt.ylabel('Energy Stored (kWh)')
plt.xlim([0,24])
plt.ylim([0,70])
plt.xticks([0,6,12,18,24])
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('total_energy_radius.pdf', bbox_inches='tight')
    
#%% Trajectory Pieces
plt.style.use(['seaborn-paper','seaborn-whitegrid'])
df = data['Winter']
dflist = []
dflist.append(df.loc[(df['time_hr']>=0)&(df['time_hr']<6.26)])
dflist.append(df.loc[(df['time_hr']>=6.26)&(df['time_hr']<8.55)])
dflist.append(df.loc[(df['time_hr']>=8.55)&(df['time_hr']<11.16)])
#dflist.append(df.loc[(df['time_hr']>=34380)&(df['time_hr']<44100)])
dflist.append(df.loc[df['time_hr']>=11.16])

for df_section in dflist:
    ax, fig = plot3DPath_NorthSouth(df_section,tight=False)
#    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('3d_path_'+str(round(df_section['time_hr'].max()))+'.pdf', bbox_inches='tight')