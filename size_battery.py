# -*- coding: utf-8 -*-
"""
"""

from __future__ import division
import numpy as np
import pandas as pd
import sys
sys.path.append("../..")
from findSteadyState_wrapper import findSteadyState
from dynamicsWrapper import uavDynamicsWrapper
from CombinedSolar import loadSmartsData
from scipy.integrate import odeint
from scipy.optimize import minimize
import time as tm
import datetime

def battery_gap(m_battery,x0,h_0,config,smartsData,m0,mb0):
    
    # Set battery mass to new value
    config['aircraft']['battery']['mass']['value'] = m_battery
    m_new = m0 + (m_battery - mb0)
    # Set total mass to new value
    config['aircraft']['mass_total']['value'] = m_new
    
    # Scale guess values
    x0 = x0 * m_battery/mb0
    
    # Solve for steady state conditions
#    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Finding Steady State')
    vmin,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config)
    print(vmin,Tpmin,alphamin,phimin,clmin,pmin)
    
    if config['trajectory']['initial_direction'] == 'Counterclockwise': # Otherwise, it does clockwise
        phimin = -phimin # Required in order to turn counterclockwise in absolute NE position, which is clockwise in xy
        
    # MV contains steady state inputss
    MV = [Tpmin,alphamin,phimin]
    
    ## Preliminary Calcs
    E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
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
    
    ## Integrate
#    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Simulating...')
    sol,output = odeint(uavDynamicsWrapper, SV0, t, args=(MV,h_0,[],smartsData,config,1),full_output=True)
    
    solData = pd.DataFrame(sol,columns=('v','gamma','psi','h','x','y','E_Batt'))
    
    # Energy Gap
    gap = abs(E_batmax - solData.E_Batt.max())
    
    print('Input: ' + str(m_battery) + ' Gap: ' + str(gap))
    
    return gap

def battery_final(m_battery,x0,h_0,config,smartsData,m0,mb0):
    
    # Set battery mass to new value
    config['aircraft']['battery']['mass']['value'] = m_battery
    m_new = m0 + (m_battery - mb0)
    # Set total mass to new value
    config['aircraft']['mass_total']['value'] = m_new
    
    # Scale guess values
    x0 = x0 * m_battery/mb0
    
    # Solve for steady state conditions
#    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Finding Steady State')
    vmin,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config)
    print(vmin,Tpmin,alphamin,phimin,clmin,pmin)
    
    if config['trajectory']['initial_direction'] == 'Counterclockwise': # Otherwise, it does clockwise
        phimin = -phimin # Required in order to turn counterclockwise in absolute NE position, which is clockwise in xy
        
    # MV contains steady state inputss
    MV = [Tpmin,alphamin,phimin]
    
    ## Preliminary Calcs
    E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
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
    
    ## Integrate
#    print('{:%H:%M:%S}'.format(datetime.datetime.now()) + ' Simulating...')
    sol,output = odeint(uavDynamicsWrapper, SV0, t, args=(MV,h_0,[],smartsData,config,1),full_output=True)
    
    solData = pd.DataFrame(sol,columns=('v','gamma','psi','h','x','y','E_Batt'))
    
    # Energy Gap
#    gap = abs(E_batmax - solData.E_Batt.max())
    final = solData.E_Batt.iloc[-1]
    
    print('Input: ' + str(m_battery) + ' Final: ' + str(final))
    
    return -final

def size_battery(config):
    ## Find Steady State / Equilibrium conditions

    # Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad) (could put in config file)
    x0 = [config['trajectory']['tp']['ss_initial_guess'],
          config['trajectory']['alpha']['ss_initial_guess'],
          abs(config['trajectory']['phi']['ss_initial_guess'])] # Note: there is an absolute value here.
    
    # Height (m)
    h_0 = config['trajectory']['h']['initial_value'] # 18288
    
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
    
    m_battery_guess = config['aircraft']['battery']['mass']['value']
    m_guess = config['aircraft']['mass_total']['value']
    
    print('Optimizing battery size')
    sol = minimize(battery_gap,
                   [m_battery_guess],
                   args=(x0,h_0,config,smartsData,m_guess,m_battery_guess),
                   method='Nelder-Mead',
                   options={'disp':True})
    if(sol.success==True):
        print('Succesful optimization')
    
    m_battery_opt = sol.x
    gap_opt = battery_gap(m_battery_opt,x0,h_0,config,smartsData,m_guess,m_battery_guess)
    print('M: ' + str(m_battery_opt))
    print('Gap: ' + str(gap_opt))
    
    
    return m_battery_opt, gap_opt

def size_battery_final(config):
    ## Find Steady State / Equilibrium conditions

    # Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad) (could put in config file)
    x0 = [config['trajectory']['tp']['ss_initial_guess'],
          config['trajectory']['alpha']['ss_initial_guess'],
          abs(config['trajectory']['phi']['ss_initial_guess'])] # Note: there is an absolute value here.
    
    # Height (m)
    h_0 = config['trajectory']['h']['initial_value'] # 18288
    
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
    
    m_battery_guess = config['aircraft']['battery']['mass']['value']
    m_guess = config['aircraft']['mass_total']['value']
    
    print('Optimizing battery size')
    sol = minimize(battery_final,
                   [m_battery_guess],
                   args=(x0,h_0,config,smartsData,m_guess,m_battery_guess),
                   method='Nelder-Mead',
                   options={'disp':True})
    if(sol.success==True):
        print('Succesful optimization')
    
    m_battery_opt = sol.x
    final_opt = battery_final(m_battery_opt,x0,h_0,config,smartsData,m_guess,m_battery_guess)
    print('M: ' + str(m_battery_opt))
    print('Final: ' + str(final_opt))
    
    
    return m_battery_opt, final_opt

if __name__ == '__main__':
    start = tm.time()
    m_battery_opt, gap_opt =size_battery_final(config)
    end = tm.time() - start
    print('Time: ' + str(end))