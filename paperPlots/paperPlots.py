# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../Sandbox/FFT/")
from plotting import plot3DPath, plotSolar, plotTotalEnergy, plot2DPath_Labeled, plot3DPath_NorthSouth, plot2DPath_Radius
from findOrbits import find_orbits
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from dashboard import fix_units
import yaml

plt.style.use(['seaborn-notebook','seaborn-whitegrid'])
plt.rc("font", family="serif")
#plt.rc('text', usetex=False)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
width  = 5
height = width / 1.618

plt.close('all')

#%%  Data
winter_folder = 'hale_2018_01_11_14_21_53 - Winter MV Dcost 15 Gamma 5'
summer_folder = 'hale_2018_01_16_14_45_58 - Summer MV Dynamic Dcost Gamma 5'
spring_folder = 'hale_2018_01_11_14_22_23 - Spring MV Dcost 15 Gamma 5'
fall_folder = 'hale_2018_01_11_14_22_13 - Fall MV Dcost 15 Gamma 5'

data={}
data['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/'+winter_folder+'/Intermediates/*.xlsx')[-1]))
data['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/'+summer_folder+'/Intermediates/*.xlsx')[-1]))
data['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/'+spring_folder+'/Intermediates/*.xlsx')[-1]))
data['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/'+fall_folder+'/Intermediates/*.xlsx')[-1]))

ss={}
ss['Winter'] = fix_units(pd.read_excel(glob.glob('../Data/'+winter_folder+'/ss*.xlsx')[-1]))
ss['Summer'] = fix_units(pd.read_excel(glob.glob('../Data/'+summer_folder+'/ss*.xlsx')[-1]))
ss['Spring'] = fix_units(pd.read_excel(glob.glob('../Data/'+spring_folder+'/ss*.xlsx')[-1]))
ss['Fall'] = fix_units(pd.read_excel(glob.glob('../Data/'+fall_folder+'/ss*.xlsx')[-1]))

rad = {}
rad['6000'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_10_15_03_41 - Winter Radius 6000/Intermediates/*.xlsx')[-1]))
rad['12000'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_10_15_03_50 - Winter Radius 12000/Intermediates/*.xlsx')[-1]))
rad['1500'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_10_15_03_59 - Winter Radius 1500/Intermediates/*.xlsx')[-1]))

SMS = {}
SMS['D1'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_13_26_54 - SM Summer D 1/ss*.xlsx')[-1]))
SMS['D2'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_04_53 - SM Summer D 2/ss*.xlsx')[-1]))
SMS['D3'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_15_00 - SM Summer D 3/ss*.xlsx')[-1]))

#SMF = {}
#SMS['D1'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_29_27 - SM Fall D 1/ss*.xlsx')[-1]))
#SMS['D2'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_29_10 - SM Fall D 2/ss*.xlsx')[-1]))
#SMS['D3'] = fix_units(pd.read_excel(glob.glob('../Data/hale_2018_01_19_14_28_48 - SM Fall D 3/ss*.xlsx')[-1]))

yaml_file = glob.glob('../Data/'+winter_folder+'/config*')[0]
with open(yaml_file, 'r') as ifile:
        config = yaml.load(ifile)

#%% Solar Plots
fig = plt.figure()
for season in data:
    df = data[season]
    df = df.loc[df['flux']>0.01]
    
    plt.plot(df.t/3600,df.flux,label=season)
    
plt.ylim([0,1400])
plt.title('Total Solar Flux Available')
plt.xlabel('Time (Hr)')
plt.ylabel('Available Flux (W/m$^2$)')
plt.legend()
plt.tight_layout()
fig.set_size_inches(width, height)
#plt.savefig('available_flux.pdf', bbox_inches='tight')

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
    
for season in data:
    df = data[season]
    df_ss = ss[season]
    
    # Total Energy and Battery Energy
    fig = plt.figure()
#    plt.plot(df.time_hr,df.te_kwh,':',color='k',label='Optimized Total Energy')
#    plt.plot(df.time_hr,df.e_batt_kwh,'-',color="0.5",label='Optimized Battery Energy')
#    plt.plot(df_ss.time_hr,df_ss.te_kwh,'-',color="k",label='Circular Total Energy')
#    plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,'-.',color="0.5",label='Circular Battery Energy')
    plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy')
    plt.plot(df.time_hr,df.e_batt_kwh,'--',label='Optimized Battery Energy')
#    plt.plot(df_ss.time_hr,df_ss.te_kwh,'-',label='SS Total Energy')
    plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,'-.',label='SS Battery Energy')
    plt.legend(loc='best')
    plt.title('Energy Storage (' + season + ')')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Energy Stored (kWh)')
    plt.xlim([0,24])
    plt.ylim([0,60])
    plt.xticks([0,6,12,18,24])
    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('total_energy_'+season+'.pdf', bbox_inches='tight')
    
#%% State Machine total energy
fig = plt.figure()
df = data['Summer']
df_ss = ss['Summer']
plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy')
plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,'-.',label='SS Battery Energy')
df = SMS['D1']
plt.plot(df.time_hr,df.te_kwh,label='State Machine 1 Total Energy')
df = SMS['D2']
plt.plot(df.time_hr,df.te_kwh,label='State Machine 2 Total Energy')
df = SMS['D3']
plt.plot(df.time_hr,df.te_kwh,label='State Machine 3 Total Energy')
plt.legend(loc='best')
plt.title('State Machine Energy Storage Comparison (Summer)')
plt.xlabel('Time (Hr)')
plt.ylabel('Energy Stored (kWh)')
plt.xlim([0,24])
plt.xticks([0,6,12,18,24])
plt.ylim([0,60])
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('total_energy_state_machine.pdf', bbox_inches='tight')
    
    
#%% Total Energy Table
df_table = pd.DataFrame(columns=['season','trajectory','max','final','change'])
for season in data:
    df_ss = ss[season]
    row = pd.Series([season,'Steady State',df_ss.te_kwh.max(),df_ss.te_kwh.tail(1).iloc[0],''],['season','trajectory','max','final','change'])
    df_table =df_table.append([row],ignore_index=True)
    df = data[season]
    change = round(df.te_kwh.tail(1).iloc[0]-df_ss.te_kwh.tail(1).iloc[0],2)
    row = pd.Series([season,'Optimized',df.te_kwh.max(),df.te_kwh.tail(1).iloc[0],change],['season','trajectory','max','final','change'])
    df_table = df_table.append([row],ignore_index=True)
    
df_table.columns = ['Season','Trajectory', 'Max Total Energy (kWh)','Final Total Energy (kWh)','Change (kWh)']
df_table = df_table.round(2)
print('Total Energy Table')
print(df_table.to_latex(index=False))

#%% Total Energy Bar Chart
fig, ax = plt.subplots()
xlabels = []
for season in data:
    xlabels.append(season)
index = np.arange(len(xlabels))
barwidth = 0.20 
    
plt.bar(index,df_table.loc[(df_table['Trajectory']=='Steady State')]['Max Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='SS Max Total Energy')
plt.bar(index+barwidth,df_table.loc[(df_table['Trajectory']=='Optimized')]['Max Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='Opt Max Total Energy')
plt.bar(index+barwidth*2,df_table.loc[(df_table['Trajectory']=='Steady State')]['Final Total Energy (kWh)'],
        width = barwidth,
        alpha = 0.8,
        label='SS Final Total Energy')
plt.bar(index+barwidth*3,df_table.loc[(df_table['Trajectory']=='Optimized')]['Final Total Energy (kWh)'],
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

#%%
# Power Required
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

plt.style.use(['seaborn-paper','seaborn-whitegrid'])

fig, axarr = plt.subplots(2, 2)
season = 'Winter'
df = data[season]
axarr[0, 0].plot(df.time_hr,df.p_n)
axarr[0, 0].set_title(season)
axarr[0, 0].set_ylabel('Power Required (W)')
season = 'Spring'
df = data[season]
axarr[0, 1].plot(df.time_hr,df.p_n)
axarr[0, 1].set_title(season)
season = 'Summer'
df = data[season]
axarr[1, 0].plot(df.time_hr,df.p_n)
axarr[1, 0].set_title(season)
axarr[1, 0].set_xlabel('Time (hr)')
axarr[1, 0].set_ylabel('Power Required (W)')
season = 'Fall'
df = data[season]
axarr[1, 1].plot(df.time_hr,df.p_n)
axarr[1, 1].set_title(season)
axarr[1, 1].set_xlabel('Time (hr)')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('power_required.pdf', bbox_inches='tight')

# Power in - Power Out
fig, axarr = plt.subplots(2, 2)
season = 'Winter'
df = data[season]
axarr[0, 0].plot(df.time_hr,df.pinout)
axarr[0, 0].set_title(season)
axarr[0, 0].set_ylabel('Power In - Out (W)')
season = 'Spring'
df = data[season]
axarr[0, 1].plot(df.time_hr,df.pinout)
axarr[0, 1].set_title(season)
season = 'Summer'
df = data[season]
axarr[1, 0].plot(df.time_hr,df.pinout)
axarr[1, 0].set_title(season)
axarr[1, 0].set_xlabel('Time (hr)')
axarr[1, 0].set_ylabel('Power In - Out (W)')
season = 'Fall'
df = data[season]
axarr[1, 1].plot(df.time_hr,df.pinout)
axarr[1, 1].set_title(season)
axarr[1, 1].set_xlabel('Time (hr)')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.tight_layout()
fig.set_size_inches(width, height)
plt.savefig('power_inout.pdf', bbox_inches='tight')

plt.style.use(['seaborn-notebook','seaborn-whitegrid'])

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
for season in data:
    df = data[season]
    x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(df)
    df = df.loc[(df['time']>=t_orbits[-2]) & (df['time']<=t_orbits[-1])]
    plot2DPath_Labeled(df,season,config)
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

#%% Full Orbits
plt.style.use(['seaborn-notebook','seaborn-whitegrid'])
for season in data:
    ax, fig = plot3DPath_NorthSouth(data[season],tight=True)
#    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('3d_path_'+season+'.pdf', bbox_inches='tight')
    
# State Machine
ax, fig = plot3DPath_NorthSouth(SMS['D1'],tight=True)
fig.set_size_inches(width, height)
plt.savefig('3d_path_StateMachine.pdf', bbox_inches='tight')

plt.style.use(['seaborn-notebook','seaborn-whitegrid'])

#%% Steady State Orbit
df = ss['Winter']
ax, fig = plot3DPath_NorthSouth(df,tight=False)
fig.set_size_inches(width, height)
plt.savefig('ss_path_winter.pdf', bbox_inches='tight')

#%% Radius Plots
r6 = pd.read_excel('rad6000.xlsx')
r15 = pd.read_excel('rad1500.xlsx')
r12 = pd.read_excel('rad12000.xlsx')
plot2DPath_Radius(r6,6000)
plot2DPath_Radius(r15,1500)
plot2DPath_Radius(r12,12000)
#for radius in rad:
#    df = rad[radius]
#    x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(df)
#    df = df.loc[(df['time']>=t_orbits[-4]) & (df['time']<=t_orbits[-3])]
#    plot2DPath_Radius(df,radius)
    
#%% Trajectory Pieces
df = data['Winter']
dflist = []
dflist.append(df.loc[(df['time']>=0)&(df['time']<21600)])
dflist.append(df.loc[(df['time']>=21600)&(df['time']<30780)])
dflist.append(df.loc[(df['time']>=30780)&(df['time']<34770)])
dflist.append(df.loc[(df['time']>=34770)&(df['time']<38160)])
dflist.append(df.loc[df['time']>=38160])

for df_section in dflist:
    ax, fig = plot3DPath_NorthSouth(df_section,tight=False)
#    plt.tight_layout()
    fig.set_size_inches(width, height)
    plt.savefig('3d_path_'+str(df_section['time'].max())+'.pdf', bbox_inches='tight')