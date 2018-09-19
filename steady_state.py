# -*- coding: utf-8 -*-

import time as tm
import numpy as np
import yaml
import datetime
import pandas as pd
import os
from scipy.integrate import odeint

from solar_functions import loadSmartsData

from findSteadyState_wrapper import findSteadyState
from dynamicsWrapper import uavDynamicsWrapper

def integrate_steady_state(config):
    '''
    Finds minimum power turn conditions, and integrates forward through the day
    '''
    
    # %% Initialization
    print('Calculating Steady-state Orbit\n')

    start_time = tm.time()

    # Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad)
    x0 = [config['trajectory']['tp']['ss_initial_guess'],
          config['trajectory']['alpha']['ss_initial_guess'],
          config['trajectory']['phi']['ss_initial_guess']] # Note: Removed and absolute value here which could have unintended consequences
    
    # Initial Height (m)
    h_0 = config['trajectory']['h']['initial_value']
    
    ## Pre-Load solar data for the day
    solar_data = config['solar']['solar_data'] 
    lat = config['solar'][solar_data]['latitude'] 
    lon = config['solar'][solar_data]['longitude'] 
    elevation = config['solar'][solar_data]['elevation'] 
    altitude = config['solar'][solar_data]['altitude'] 
    year = config['solar'][solar_data]['year'] 
    month = config['solar'][solar_data]['month'] 
    day = config['solar'][solar_data]['day'] 
    zone = config['solar'][solar_data]['zone'] 
    smartsData = loadSmartsData(lat,lon,elevation, altitude, year,
                  month, day, zone)
    
    #%% Solve for steady state conditions
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Finding Steady State')
    vmin,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config)
    
    #%% Prepare for integration
    
    # MV contains steady state inputs
    MV = [Tpmin,alphamin,phimin]
    
    ## Preliminary Calcs
    E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
    m_battery = config['aircraft']['battery']['mass']['value'] # 212.0 # (kg) (FB)
    E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
    
    # Initial Conditions
    v_0 = vmin # 43.994 # Initial Velocity (m)
    gamma_0 = config['trajectory']['gamma']['initial_value'] # 0 # Initial flight path angle (rad)
    psi_0 = config['trajectory']['psi']['initial_value'] # 0 # Initial heading (rad)
    x0 = config['trajectory']['x']['initial_value'] # 0 # Horizontal distance (m)
    y0 = config['trajectory']['y']['initial_value'] # -3000 # Other horizontal distance
    initial_SOC = config['aircraft']['battery']['initial_state_of_charge']
    E_Batt_0 = E_batmax*initial_SOC # *0.20 # Initial Battery Charge
    
    # Put initial conditions all together in SV0
    SV0 = [v_0,gamma_0,psi_0,h_0,x0,y0,E_Batt_0] 
    
    # Setup time range for integration
    startTime = config['optimization']['start_time']['value'] # 26160/3600.0 # Hours
    timeStep = config['optimization']['time_step']['value'] # 60 # seconds
    endTime = config['optimization']['end_time']['value'] # 26160/3600.0 + 24 # Hours
    t = np.arange(startTime*3600,startTime+3600*endTime+timeStep,timeStep) # t must be in seconds
    
    #%% Integrate
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Simulating...')
    sol,output = odeint(uavDynamicsWrapper, SV0, t, args=(MV,h_0,[],smartsData,config,1),full_output=True)
    
    #%% Process integrated solution
    # Put solution in dataframe
    solData = pd.DataFrame(sol,columns=('v','gamma','psi','h','x','y','E_Batt'))
    
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Recovering Intermediates...')
    # Post process to recover the intermediates we want
    solData['time'] = t
    model_outputs = [pd.Series(uavDynamicsWrapper(SV[1],[],MV,h_0,[],smartsData,config,5)).to_frame() for SV in solData.iterrows()]
    intermediates = pd.concat(model_outputs,axis=1,ignore_index=True).transpose()
        
    # Combined intermediates with integrated data
    solData = pd.concat([solData,intermediates],axis=1)
    
    # Add in time and MVs
    solData['time'] = t - startTime*3600 # 28440/3600.0 # Included in config file
    solData['t'] = t
    solData['tp'] = MV[0]
    solData['alpha'] = MV[1]
    solData['phi'] = MV[2]
    solData['e_batt'] = solData['E_Batt']
    solData['e_batt_max'] = E_batmax
    
    # Save out data for MPC
    time_stamp = config['file']['time_stamp']
    solDataOut = solData[['time', 'flux', 't','zenith','azimuth','sn1','sn2','sn3','sun_h']]
    filename = os.path.join(config['file']['results_folder'],'apm_input_' + time_stamp + '.csv')
    solDataOut.to_csv(filename, index=False)
    
    # Save out steady state solution
    simDataOut = solData[['time', 'tp', 'phi', 'theta', 'alpha', 'gamma', 'psi', 'v', 'x', 'y', 'h', 'dist', 'te', 'e_batt', 'e_batt_max', 'p_bat', 'p_n', 'p_solar', 'panel_efficiency',
                          'd', 'cd', 'c_d_p', 'cl', 'rho', 're', 'm', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3']]
    
    filenameSim = filename = os.path.join(config['file']['results_folder'],'ss_results_' + str(time_stamp) + '_test' + '.xlsx')
    simDataOut.to_excel(filenameSim, index=False)
    
    # Update initial values for optimization in config file to be updated in model file
    config['trajectory']['tp']['initial_value'] = float(simDataOut['tp'][0])
    config['trajectory']['alpha']['initial_value'] = float(simDataOut['alpha'][0])
    config['trajectory']['phi']['initial_value'] = float(simDataOut['phi'][0])
    config['trajectory']['v']['initial_value'] = float(simDataOut['v'][0])
    
    
    # Update configuration file and over-write the old one
    with open(config['file']['config_filepath'], 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    # Print timing results
    end = tm.time()
    solveTime = end - start_time
    print("Solve Time: " + str(solveTime))