# -*- coding: utf-8 -*-
"""
Updates:
    7/12/17:
        - Added error checking with good_solution to determine when APMonitor solves successfully.
        - Changed good_sol variable to sol_returned to better reflect what it indicates
        - Added date and timestamp to be saved for each iteration
        - Added iteration_save_frequency and configured the script to save data this often
    7/14/17:
        - Added error handling to trigger a resolve with coldstart for a failed solution
    7/18/17:
        - Added max_iterations as defined in config file
        - Fixed bug that was causing files to remain open.
        - Saves the number of iterations for each horizon (there's still a bug in this, but will be investigated)
        - Saves the objective function value ("objfcnval") for each horizon
    8/10/17:
        - Merged with Abe's pandas dataframe changes - still needs testing (and plotting script)
    8/29/17:
        - Configured to work correctly with Pandas change
        - Matched the column names between SS and Opt
    9/26/17:
        - Made a lot of changes - see commit comment
        - Added a variable cd to the model files which is equal to C_D so it will pull that data correctly
"""

# Import APM package
#from APMonitor import apm, apm_load, csv_load, apm_option, apm_info, apm_web, apm_sol
from apm import *
import numpy as np
import pandas as pd
import time as tm
import datetime
import os
import yaml
from plotting import plot3DPath, plotSolar, plotTotalEnergy # Need to get this module from Abe
from status import updateStatus
import sys
import shutil
import glob
from plotly.offline import plot
import plotly.graph_objs as go

def optimize_MPC(m,config,iters):
    #%% Import configuration file - Choose automatic or manual mode
    
    # Automatic = use config file stored in memory
    # Manual = import previously defined config file by specifying folder name
    
    mode = 'automatic' # 'manual', 'automatic' 
    
    if mode=='automatic': ##### Need to finish updating this
        
        ### Import configuration file
        # Run this code after defining the config file (and editing it if needed), with 
        # the variables still in memory.
        
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
        print('Manual Mode Selected')
        
        ### Import a configuration file manually, from a specified folder/name
        
        folder_name = 'hale_2017_11_14_16_38_45 - Tsh 20 H 110'  # Folder that contains yaml file
        time_stamp = folder_name.split()[0][5:]
        file_name = 'config_file_' + time_stamp + '.yml'
        cwd = os.getcwd()
        directory = cwd+'/Data/'+folder_name+'/'
        with open(directory+file_name, 'r') as ifile:
            config_file = yaml.load(ifile)
        config = config_file
        newpath = config['file']['new_path']
        oldpath = config['file']['original_path']
    
    else:
        print('Error: Choose manual or automatic mode.')
    
    # Need above code to define model file and save it in the folder, and call it below
    
    
    #%%
    
    # Change to correct directory to import model and csv files, and save the data
    os.chdir(newpath)
    
    # Set timestep
    step = config['optimization']['time_step']['value']
    
    # Load solar data for whole day
    filename = 'apm_input_' + time_stamp + '.csv'
    dayDataPart = pd.read_csv(filename,delimiter=',')
    
    filename = 'time_mpc_' + str(step) + 's.csv'
    
    # Global Options
    m.options.max_iter = config['optimization']['max_iterations']
    m.options.cv_type = 1
    m.options.time_shift = config['optimization']['time_shift']
    time_shift = config['optimization']['time_shift']
    m.options.csv_read = 2
    m.options.nodes = 2
    m.options.reduce = 4
    m.options.web = 0
    m.options.ctrl_units = 1
    m.options.solver = 3
    m.options.imode = 6
    m.options.otol = 1e-4
    m.options.rtol = 1e-4
#    m.solver_options = ['bound_frac 0.0001',
#                        'bound_push 0.0001',
#                        'slack_bound_frac 0.0001',
#                        'slack_bound_push 0.0001',
#                        'bound_mult_init_method mu-based']
#    
    # Setup Variables
    m.alpha.status = 1
    m.alpha.fstatus = 0
    m.alpha.lower = config['trajectory']['alpha']['min']
    m.alpha.upper = config['trajectory']['alpha']['max']
    m.alpha.dmax = config['trajectory']['alpha']['dmax']
    m.alpha.dcost = config['trajectory']['alpha']['dcost']
    m.phi.status = 1
    m.phi.fstatus = 0
    m.phi.lower = config['trajectory']['phi']['min']
    m.phi.upper = config['trajectory']['phi']['max']
    m.phi.dmax = config['trajectory']['phi']['dmax']
    m.phi.dcost = config['trajectory']['phi']['dcost']
    m.phi.cost = config['trajectory']['phi']['cost']
    m.tp.status = 1
    m.tp.fstatus = 0
    m.tp.lower = config['trajectory']['tp']['min']
    m.tp.upper = config['trajectory']['tp']['max']
    m.tp.dmax = config['trajectory']['tp']['dmax']
    m.tp.dcost = config['trajectory']['tp']['dcost']
    m.p_bat.status = 1
    m.p_bat.fstatus = 0
#    m.mu_slack.status = 1
#    m.mu_slack.fstatus = 0
    
    #CVs
#    m.h.status = 1
#    m.h.sphi = config['trajectory']['h']['max']
#    m.h.splo = config['trajectory']['h']['min']
##    m.h.wsphi = 100
##    m.h.wsplo = 100
#    m.h.tr_init = 0
#    m.dist.status = 1
#    m.dist.sphi = config['trajectory']['x']['max']
#    m.dist.splo = 0
##    m.dist.wsphi = 100
#    m.dist.wsplo = 0
#    m.dist.tr_init = 0
    m.dist.status = 1
    m.dist.sphi = config['trajectory']['x']['max']
    m.dist.splo = 0
    m.dist.wsphi = 100
    m.dist.wsplo = 0
    m.dist.tr_init = 0
    m.cl.status = 1
    m.cl.sphi = 1.5
    m.cl.splo = 0
    m.cl.wsphi = 10000
    m.cl.wsplo = 0
    m.cl.tr_init = 0
    
    
    
    # Initialize storage from first row of steady state solution
    filenameSim = 'ss_results_' + str(time_stamp) + '_test' + '.xlsx'
    ss_data = pd.read_excel(filenameSim)
    dataOut = ss_data.head(1)
    
    # Use steady state values for initial guess
    if(config['wind']['use_wind']==False):
        columns = ['time', 'flux','t','zenith','azimuth','sn1','sn2','sn3',
                   'tp', 'phi', 'alpha', 'gamma', 'psi', 'v', 'x', 'y', 'h','dist','te','e_batt']
    else:
        columns = ['time', 'flux','t','zenith','azimuth','sn1','sn2','sn3',
                   'tp', 'phi', 'alpha', 'gamma','chi', 'v_g', 'v_a', 'x', 'y', 'h','dist','te','e_batt']
    ss_data = ss_data[columns]
    
    # Load loop parameters
    horizon_length = config['optimization']['horizon_length']
    time_step = config['optimization']['time_step']['value']
    start_time = config['optimization']['start_time']['value']
    end_time = config['optimization']['end_time']['value']
    
    length = int(3600*(end_time - start_time)/time_step - horizon_length)
    
    save_freq = config['optimization']['iteration_save_frequency']
    
    error_flag = 0 # Initialize flag
    
    # Begin timing
    start = tm.time()
    
    # Set horizon
    m.time = ss_data['time'].iloc[0:horizon_length+1].values
    
    storeIter = np.zeros([horizon_length+1,3,500])
    i = 0
    for j in range(0,iters):
        print('*************************************')
        print('*************************************')
        print('')
        print('Max Iterations: ' +str(i))
        print('')
        print('*************************************')
        print('*************************************')
        
        m.options.max_iter = j
        
        iter_start = tm.time()
#        if(config['wind']['use_wind']==False):
#            if(i<250):
#                m.tp.dcost = config['trajectory']['tp']['dcost']*(1/((i+1)/250))
#                m.phi.dcost = config['trajectory']['phi']['dcost']*(1/((i+1)/250))
#                m.alpha.dcost = config['trajectory']['alpha']['dcost']*(1/((i+1)/250))
#            else:
#                m.tp.dcost = config['trajectory']['tp']['dcost']
#                m.phi.dcost = config['trajectory']['phi']['dcost']
#                m.alpha.dcost = config['trajectory']['alpha']['dcost']
#        if(i==0):
        df1 = ss_data.iloc[0:horizon_length+1,:]
        m.flux.value = df1['flux'].values
        m.t.value = df1['t'].values
        m.zenith.value = df1['zenith'].values
        m.azimuth.value = df1['azimuth'].values
        m.sn1.value = df1['sn1'].values
        m.sn2.value = df1['sn2'].values
        m.sn3.value = df1['sn3'].values
        m.tp.value = df1['tp'].values
        m.phi.value = df1['phi'].values
        m.alpha.value = df1['alpha'].values
        m.gamma.value = df1['gamma'].values
        if(config['wind']['use_wind']):
            m.chi.value = df1['chi'].values
            m.v_g.value = df1['v_g'].values
            m.v_a.value = df1['v_a'].values
        else:
            m.psi.value = df1['psi'].values
            m.v.value = df1['v'].values
        m.x.value = df1['x'].values
        m.y.value = df1['y'].values
        m.h.value = df1['h'].values
        m.dist.value = df1['dist'].values
        m.e_batt.value = df1['e_batt'].values
        m.te.value = df1['te'].values
        m.p_total.value = np.zeros(len(df1))
#        else:
#            df1 = dayDataPart.iloc[i:i+horizon_length+1,:]
#            m.flux.value = df1['flux'].values
#            m.t.value = df1['t'].values
#            m.zenith.value = df1['zenith'].values
#            m.azimuth.value = df1['azimuth'].values
#            m.sn1.value = df1['sn1'].values
#            m.sn2.value = df1['sn2'].values
#            m.sn3.value = df1['sn3'].values
            
        m.solve()
        
        success_check = m.options.appstatus
        
        if(success_check==1):
            # Then grab extra data from the server
            objfcnval = m.options.objfcnval
            iterations = 0#[int(s) for s in [s for s in solver_output if "Iterations" in s][0].split() if s.isdigit()][0]
        else:
            print('No successful solution found.')
            iterations = 0#[int(s) for s in [s for s in solver_output if "Iterations" in s][0].split() if s.isdigit()][0]  
            objfcnval = 0 
            
        sol = m.load_results()
        solData = pd.DataFrame.from_dict(sol) # convert APM solution to dataFrame
        
        storeIter[:,0,j] = solData['x'].values
        storeIter[:,1,j] = solData['y'].values
        storeIter[:,2,j] = solData['h'].values
        
        if(success_check==1):
            break
        else:
            j = j + 1
    
    data = []
    for k in range(j):
        trace = go.Scatter3d(
            x=storeIter[:,0,k], y=storeIter[:,1,k], z=storeIter[:,2,k],
            marker=dict(
                size=2,
#                color=storeIter[:,2,0],
#                colorscale='Viridis',
            ),
            line=dict(
                color='#1f77b4',
                width=5
            )
        )
        data.append(trace)
        
    slider_values = [] # Initialize array
    for i in range(len(data)):
        # Hide all the traces
        mask = dict(
            method = 'restyle',
            args = ['visible', [False] * len(data)],
        )
        # Except the one we're looking at
        mask['args'][1][i] = True
        # Set slider value at this point equal to our mask
        slider_values.append(mask)
    # Now we create the actual slider
    sliders = [dict(
        active = 0, # Slider starts at 0'th value
        steps = slider_values # Set slider steps equal to our array of values
    )]
    # Create figure layout
    layout = dict(sliders=sliders,
                  scene = dict(
                    xaxis = dict(
                        range = [storeIter[:,0,:j].min(),storeIter[:,0,:j].max()],),
                    yaxis = dict(
                        range = [storeIter[:,1,:j].min(),storeIter[:,1,:j].max()],),
                    zaxis = dict(
                        range = [18288-500,18288+500],),)
                    )
    fig = dict(data=data,layout=layout)
    plot(fig)
    
if __name__=='__main__':
    optimize_MPC(m,config,500)