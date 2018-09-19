# -*- coding: utf-8 -*-
"""
Updates:
    7/18/17 - Fixed bug that was causing files to remain open.
    8/29/17 - Added many variables for the SS simulation to match the Opt
    9/15/17 - Negated the calculated value of phimin in order to correctly do clockwise turns
    9/20/17 - Added in clockwise vs counterclockwise logic to negate phimin accordingly
"""

from __future__ import division
import numpy as np
import pandas as pd
from numpy import pi, sqrt, cos, sin, exp, tan, log10
from findSteadyState_wrapper import findSteadyState #On purpose, since we ignore wind initially in the ss
from dynamicsWrapper_wind import uavDynamicsWrapper_wind
from scipy.optimize import root
from scipy.integrate import odeint
import time as tm
from CombinedSolar import solarFlux, loadSmartsData
from plotting import plot3DPath
import datetime
import yaml
import os
import progressbar # pip install progressbar2
import sys

#%% Choose automatic or manual mode

# Automatic = use config file stored in memory
# Manual = import previously defined config file by specifying folder name

mode = 'automatic' # 'manual', 'automatic'

if mode=='automatic':
    
    ### Import configuration file
    # Run this code after defining the config file (and editing it if needed), with 
    # the variables still in memory.
    
    cwd = os.getcwd()
    newpath = config['file']['new_path']
    oldpath = config['file']['original_path']
    os.chdir(newpath)
    with open(config['file']['configuration'], 'r') as ifile:
        config_file = yaml.load(ifile)
    config = config_file
    os.chdir(oldpath)
    time_stamp = config['file']['time_stamp']
    
elif mode=='manual':
    
    str(input('Are you sure you want to use Manual mode? \n - If so, press enter. \n - If not, now is your chance to stop the code from running.'))  
    
    ### Import a configuration file manually, from a specified folder/name
    
    folder_name = 'hale_2017_09_15_16_05_10'  # Folder that contains yaml file
    time_stamp = folder_name[5:]
    file_name = 'config_file_' + time_stamp + '.yml'
    cwd = os.getcwd()
    os.chdir(cwd+'/Results/'+folder_name+'/')
    with open(file_name, 'r') as ifile:
        config_file = yaml.load(ifile)
    config = config_file
    newpath = config['file']['new_path']
    oldpath = config['file']['original_path']    
    os.chdir(cwd)

else:
    print('Error: Choose manual or automatic mode.')
    
print('Calculating Steady-state Orbit')
print()    

#%%

start = tm.time()

## Find Steady State / Equilibrium conditions

# Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad) (could put in config file)
x0 = [config['trajectory']['tp']['ss_initial_guess'],
      config['trajectory']['alpha']['ss_initial_guess'],
      abs(config['trajectory']['phi']['ss_initial_guess'])] # Note: there is an absolute value here.

# Height (m)
h_0 = config['trajectory']['h']['initial_value'] # 18288

# Wind
w_n = config['wind']['w_n']
w_e = config['wind']['w_e']
w_d = config['wind']['w_d']
w = [w_n,w_e,w_d]

## Pre-Load solar data for the day
solar_data = config['solar']['solar_data'] # <---- Still need to finish debugging this method
#    # Albuquerque NM - Winter Solstice
lat = config['solar'][solar_data]['latitude'] # 35.0853
lon = config['solar'][solar_data]['longitude'] # -106.6056
elevation = config['solar'][solar_data]['elevation'] # 1.619
altitude = config['solar'][solar_data]['altitude'] # 20
year = config['solar'][solar_data]['year'] # 2016
month = config['solar'][solar_data]['month'] # 12
day = config['solar'][solar_data]['day'] # 21
zone = config['solar'][solar_data]['zone'] # -7

smartsData = loadSmartsData(lat,lon,elevation, altitude, year,
              month, day, zone)

# Solve for steady state conditions
print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Finding Steady State')
v_a_min,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config_file)

if config['trajectory']['initial_direction'] == 'Counterclockwise': # Otherwise, it does clockwise
    phimin = -phimin # Required in order to turn counterclockwise in absolute NE position, which is clockwise in xy

#%%

# MV contains steady state inputss
MV = [Tpmin,alphamin,phimin]

## Preliminary Calcs
E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
m_battery = config['aircraft']['battery']['mass']['value'] # 212.0 # (kg) (FB)
E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)

# Initial Conditions
v_a_0 = v_a_min # 43.994 # Initial Velocity (m)
gamma_0 = config['trajectory']['gamma']['initial_value'] # 0 # Initial flight path angle (rad)
psi_0 = config['trajectory']['psi']['initial_value'] # 0 # Initial heading (rad)

## Find initial chi
#def find_chi(chi,v_g,w_n,w_e,gamma,config):
#    v_w = sqrt(w_n**2 + w_e**2 + w_d**2) # Magnitidue of wind speed
#    v_a = sqrt(v_g**2-2*v_g*(w_n*cos(chi)*cos(gamma)+w_e*sin(chi)*cos(gamma)-w_d*sin(gamma))+v_w**2) # Positive root    
#    gamma_a = np.arcsin((v_g*sin(gamma) + w_d)/v_a) #WIND
#    psi = chi - np.arcsin((-w_n*sin(chi) + w_e*cos(chi))/(v_a*cos(gamma_a))) #WIND    
#    diff = psi - config['trajectory']['psi']['initial_value']
#    return diff
#
#sol = root(find_chi,[psi_0],args=(v_a_0,w_n,w_e,gamma_0,config_file))
#if(sol.success==False):
#        print('Could not find initial chi')
#        sys.exit()
#
#chi_0 = sol.x[0]
chi_0 = psi_0
x0 = config['trajectory']['x']['initial_value'] # 0 # Horizontal distance (m)
y0 = config['trajectory']['y']['initial_value'] # -3000 # Other horizontal distance
initial_SOC = config['aircraft']['battery']['initial_state_of_charge']
E_Batt_0 = E_batmax*initial_SOC # *0.20 # Initial Battery Charge

# Put initial conditions all together in SV0
SV0 = [v_a_0,gamma_0,chi_0,h_0,x0,y0,E_Batt_0] 

# Setup time range for integration
startTime = config['optimization']['start_time']['value'] # 26160/3600.0 # Hours
timeStep = config['optimization']['time_step']['value'] # 60 # seconds
endTime = config['optimization']['end_time']['value'] # 26160/3600.0 + 24 # Hours
#endTime = startTime+0.25
t = np.arange(startTime*3600,startTime+3600*endTime+timeStep,timeStep) # t must be in seconds

## Integrate
print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Simulating...')
sol,output = odeint(uavDynamicsWrapper_wind, SV0, t, args=(MV,h_0,[],w,smartsData,config_file,1),full_output=1)

# Put solution in dataframe
solData = pd.DataFrame(sol,columns=('v_a','gamma','chi','h','x','y','E_Batt'))

#%%
print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Recovering Intermediates...')
# Post process to recover the intermediates we want
intermediates = pd.DataFrame()
bar = progressbar.ProgressBar()
for i, time in enumerate(bar(t)):
    SV = solData.iloc[i,:]
    model_output = uavDynamicsWrapper_wind(SV,time,MV,h_0,[],w,smartsData,config_file,5)
    model_output_df = pd.DataFrame(model_output[0],index=[i])
    intermediates = pd.concat([intermediates, model_output_df])
    
# Combined intermediates with integrated data
solData = pd.concat([solData,intermediates],axis=1)

# Add in time and MVs
solData['time'] = t - startTime*3600 # 28440/3600.0 # Included in config file
solData['t'] = t
#solData['tp'] = MV[0]
#solData['alpha'] = MV[1]
#solData['phi'] = MV[2]
solData['e_batt'] = solData['E_Batt']
solData['w_n'] = w_n
solData['w_e'] = w_e
solData['w_d'] = w_d


# Save out data for MPC
solDataOut = solData[['time', 'flux', 't','zenith','azimuth','sn1','sn2','sn3','sun_h']]

os.chdir(newpath) # Save the data in the correct folder
filename = 'apm_input_' + time_stamp + '.csv' # Change this name to be more descriptive
solDataOut.to_csv(filename, index=False)

# Save out steady state solution
simDataOut = solData[['time', 'tp', 'phi', 'theta', 'alpha', 'beta', 'gamma', 'gamma_a', 'psi', 'v_a_angle', 'chi', 'v_g', 'v_a', 'x', 'y', 'h', 'dist', 'te', 'e_batt', 'p_bat', 'p_n', 'p_solar', 'panel_efficiency',
                      'd', 'cd', 'cl', 'rho', 're', 'm', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3','w_n','w_e','w_d']]

filenameSim = 'ss_results_' + str(time_stamp) + '_test' + '.xlsx'
simDataOut.to_excel(filenameSim, index=False)

# Update initial values for optimization in config file to be updated in model file
config['trajectory']['tp']['initial_value'] = float(simDataOut['tp'][0])
config['trajectory']['alpha']['initial_value'] = float(simDataOut['alpha'][0])
config['trajectory']['phi']['initial_value'] = float(simDataOut['phi'][0])
config['trajectory']['v']['initial_value'] = float(simDataOut['v_g'][0])

# Update configuration file and over-write the old one
os.chdir(newpath)
with open(config['file']['configuration'], 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False) # Test this
os.chdir(oldpath)

# Plot
#plot3DPath(solData)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(solData.x,solData.y,solData.h)
ax.set_xlim([solData.x[solData.x>-10000].min(),solData.x[solData.x<10000].max()])
ax.set_ylim([solData.y[solData.y>-10000].min(),solData.y[solData.y<10000].max()])
ax.set_zlim([solData.h[solData.h>1000].min()-50,solData.h[solData.h<30000].max()+50])

end = tm.time()
solveTime = end - start
print("Solve Time: " + str(solveTime))

os.chdir(oldpath)