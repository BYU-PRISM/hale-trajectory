# -*- coding: utf-8 -*-

import time as tm
import numpy as np
import datetime
import pandas as pd
import os
import sys
from scipy.integrate import odeint
from scipy.optimize import minimize

from solar_functions import loadSmartsData
from utilities import save_object

from dynamics import uavDynamics

def integrate_steady_state(config):
    '''
    Finds minimum power turn conditions, and integrates forward through the day
    '''
    
    # %% Initialization
    print('Calculating Steady-state Orbit\n')

    start_time = tm.time()

    # Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad)
    x0 = [config.aircraft.tp.ss_initial_guess,
          config.aircraft.alpha.ss_initial_guess,
          config.aircraft.phi.ss_initial_guess] # Note: Removed an absolute value here which could have unintended consequences
    
    # Initial Height (m)
    h_0 = config.h.initial_value
    
    ## Pre-Load solar data for the day
    lat = config.solar.latitude 
    lon = config.solar.longitude 
    elevation = config.solar.elevation 
    altitude = config.solar.altitude 
    year = config.solar.year 
    month = config.solar.month 
    day = config.solar.day 
    zone = config.solar.zone 
    smartsData = loadSmartsData(lat,lon,elevation, altitude, year,
                  month, day, zone)
    
    #%% Solve for steady state conditions
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Finding Steady State')
    vmin,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config)
    
    #%% Prepare for integration
    
    # MV contains steady state inputs
    MV = [Tpmin,alphamin,phimin]
    
    ## Preliminary Calcs
    E_d = config.aircraft.battery_energy_density.value # Battery energy density (W*hr/kg) 
    m_battery = config.aircraft.mass_battery.value # (kg) 
    E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
    
    # Initial Conditions
    v_0 = vmin # Initial Velocity (m)
    gamma_0 = config.aircraft.gamma.level # 0 # Initial flight path angle (rad)
    psi_0 = config.aircraft.psi.initial_value # Initial heading (rad)
    x0 = config.x.initial_value  # Horizontal distance (m)
    y0 = config.y.initial_value # Other horizontal distance
    initial_SOC = config.aircraft.battery_initial_SOC.value # Initial state of charge
    E_Batt_0 = E_batmax*initial_SOC # Initial Battery Charge
    
    # Put initial conditions all together in SV0
    SV0 = [v_0,gamma_0,psi_0,h_0,x0,y0,E_Batt_0] 
    
    # Setup time range for integration
    startTime = config.start_time.value # 26160/3600.0 # Hours
    timeStep = config.time_step.value # 60 # seconds
    endTime = config.end_time.value # 26160/3600.0 + 24 # Hours
    t = np.arange(startTime*3600,startTime+3600*endTime+timeStep,timeStep) # t must be in seconds
    
    #%% Integrate
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Simulating...')
    sol,output = odeint(uavDynamics, SV0, t, args=(MV,h_0,[],smartsData,config,1),full_output=True)
    
    #%% Process integrated solution
    # Put solution in dataframe
    solData = pd.DataFrame(sol,columns=('v','gamma','psi','h','x','y','E_Batt'))
    
    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Recovering Intermediates...')
    # Post process to recover the intermediates we want
    solData['time'] = t
    model_outputs = [pd.Series(uavDynamics(SV[1],[],MV,h_0,[],smartsData,config,5)).to_frame() for SV in solData.iterrows()]
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
    time_stamp = config.time_stamp
    solDataOut = solData[['time', 'flux', 't','zenith','azimuth','sn1','sn2','sn3','sun_h']]
    filename = os.path.join(config.results_folder,'apm_input_' + time_stamp + '.csv')
    solDataOut.to_csv(filename, index=False)
    
    # Save out steady state solution
    simDataOut = solData[['time', 'tp', 'phi', 'theta', 'alpha', 'gamma', 'psi', 'v', 'x', 'y', 'h', 'dist', 'te', 'e_batt', 'e_batt_max', 'p_bat', 'p_n', 'p_solar', 'panel_efficiency',
                          'd', 'cd', 'cl', 'rho', 're', 'm', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3']]
    
    filenameSim = filename = os.path.join(config.results_folder,'ss_results_' + str(time_stamp) + '_test' + '.xlsx')
    simDataOut.to_excel(filenameSim, index=False)
    
    # Update initial values for optimization in config file to be updated in model file
    config.aircraft.tp.initial_value = float(simDataOut['tp'][0])
    config.aircraft.alpha.initial_value = float(simDataOut['alpha'][0])
    config.aircraft.phi.initial_value = float(simDataOut['phi'][0])
    config.aircraft.v.initial_value = float(simDataOut['v'][0])
    
    # Print timing results
    end = tm.time()
    solveTime = end - start_time
    print("Solve Time: " + str(solveTime))
    
def findSteadyState(x0,h_0,smartsData,config):
    '''
    For a given height, this returns the velocity, thrust, angle of attack,
    and bank angle for a minimum power level turn with the desired radius.
    '''

#    # Initial guess for velocity
    v_0 = config.aircraft.v.ss_initial_guess # 35

    cons = ({'type': 'eq', 'fun':ss_constraints,'args':(h_0,smartsData,config)})
    
    bounds = [(config.aircraft.tp.min,config.aircraft.tp.max),
               (config.aircraft.phi.min,config.aircraft.phi.max),
               (config.aircraft.alpha.min,config.aircraft.alpha.max),
               (0,100)]
    
    # Find dynamic equlibrium
    x0v0 = np.r_[x0,v_0]
    sol = minimize(ss_objective,
                   [x0v0],
                   args=(h_0,smartsData,config),
                   bounds = bounds,
                   constraints=cons,
                   method='SLSQP',
                   options={'disp':True,'eps':1e-8,'ftol':1e-8})

    if(sol.success==False):
        print('Could not find minimum velocity')
        sys.exit()
    Tpmin = sol.x[0]
    alphamin = sol.x[1]
    phimin = sol.x[2]
    vmin = sol.x[3]
    clmin = uavDynamics(sol.x[:3],[],[],h_0,vmin,smartsData,config,4)
    pmin = sol.fun
    
    return vmin,Tpmin,alphamin,phimin,clmin,pmin

def ss_objective(xv,h_0,smartsData,config):
    x0 = xv[:3]
    v = xv[3]
    P_N = uavDynamics(x0,[],[],h_0,v,smartsData,config,3)
    return P_N

def ss_constraints(xv,h_0,smartsData,config):
    x0 = xv[:3]
    v = xv[3]
    d = uavDynamics(x0,[],[],h_0,v,smartsData,config,2)
    return d
