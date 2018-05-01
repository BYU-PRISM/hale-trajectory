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
#from apm import *
import numpy as np
import pandas as pd
import time as tm
import datetime
import os
import yaml
from plotting import plot3DPath, plotSolar, plotTotalEnergy
from status import updateStatus
import sys
import shutil
import glob

def optimize_MPC(m,config):
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
    
    
    #%% Set Optimizer Options
    
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
    m.solver_options = ['linear_solver '+config['optimization']['linear_solver']]
    
    # Setup Variables
    m.alpha.status = 1
    m.alpha.fstatus = 0
    m.alpha.lower = config['trajectory']['alpha']['min']
#    m.alpha.upper = config['trajectory']['alpha']['max']
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
    m.h.status = 1
    m.h.sphi = config['trajectory']['h']['max']
    m.h.splo = config['trajectory']['h']['min']
    m.h.wsphi = 100
    m.h.wsplo = 100
    m.h.tr_init = 0
    m.dist.status = 1
    m.dist.sphi = config['trajectory']['x']['max']
#    m.dist.sphi = 1
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
                   'tp', 'phi', 'alpha', 'gamma','chi', 'v_g', 'v_a', 'x', 'y', 'h','dist','te','e_batt','w_n','w_e','w_d']
    ss_data = ss_data[columns]
    
    # Load loop parameters
    horizon_length = config['optimization']['horizon_length']
    time_step = config['optimization']['time_step']['value']
    start_time = config['optimization']['start_time']['value']
    end_time = config['optimization']['end_time']['value']
    
    length = int(3600*(end_time - start_time)/time_step - horizon_length)
    
    save_freq = config['optimization']['iteration_save_frequency']
    
    error_flag = 0 # Initialize flag
    
    # Initialize convex weights with ss
    df_prev = ss_data.iloc[0:horizon_length+1,:]
    convex_horizon = np.ones(horizon_length)
    convex_horizon[-time_shift] = 0
    
    # Begin timing
    start = tm.time()
    
    # Set horizon
    m.time = ss_data['time'].iloc[0:horizon_length+1].values
    
    #%% MPC Loop
    for i in range(0,length,time_shift):
        
        # Initialize resolve to zero, we'll attempt to solve until we succeed or resolve
        # hits the max resolves, adjusting the options each time if needed.
        resolve = 0
        max_resolve = 3
        success_check = 0
        
        while(success_check==0 and resolve < max_resolve):
            
            print('*************************************')
            print('*************************************')
            print('')
            print('Begin Iteration: ' +str(i) + ' Re-solve: ' + str(resolve))
            print('')
            print('*************************************')
            print('*************************************')
        
            iter_start = tm.time()
            
            # Constrict MVs at start, this forces the optimizer to stay closer to the 
            # steady state solution for the first few iterations.  Seems to help convergence
            # sometimes
            if(i<250 and config['wind']['use_wind']==False):
                m.tp.dcost = config['trajectory']['tp']['dcost']*(1/((i+1)/250))
                m.phi.dcost = config['trajectory']['phi']['dcost']*(1/((i+1)/250))
                m.alpha.dcost = config['trajectory']['alpha']['dcost']*(1/((i+1)/250))
            else:
                m.tp.dcost = config['trajectory']['tp']['dcost']
                m.phi.dcost = config['trajectory']['phi']['dcost']
                m.alpha.dcost = config['trajectory']['alpha']['dcost']
                
            # For the first iteration, load the steady state as an initial guess
            if(i==0):
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
                    m.w_n.value = df1['w_n'].values
                    m.w_e.value = df1['w_e'].values
                    m.w_d.value = df1['w_d'].values
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
                # Load previous solution to convexify
                m.x_prev = df_prev['x'].values
                m.y_prev = df_prev['y'].values
                m.h_prev = df_prev['h'].values
                m.gamma_prev = df_prev['gamma'].values
                m.convex_weight = np.ones(len(df_prev['x'].values))*1e-6
                m.convex_horizon = np.ones(len(df_prev['x'].values))
            # Otherwise just load in the new solar data
            else:
                df1 = dayDataPart.iloc[i:i+horizon_length+1,:]
                m.flux.value = df1['flux'].values
                m.t.value = df1['t'].values
                m.zenith.value = df1['zenith'].values
                m.azimuth.value = df1['azimuth'].values
                m.sn1.value = df1['sn1'].values
                m.sn2.value = df1['sn2'].values
                m.sn3.value = df1['sn3'].values
                if(config['wind']['use_wind']):
                    m.w_n.value = df1['w_n'].values
                    m.w_e.value = df1['w_e'].values
                    m.w_d.value = df1['w_d'].values
                # Load previous solution to convexify
                m.x_prev = df_prev['x'].values
                m.y_prev = df_prev['y'].values
                m.h_prev = df_prev['h'].values
                m.convex_weight = np.ones(len(df_prev['x'].values))*1e-6
                m.convex_horizon = convex_horizon
                
            # Adjust settings for progressive resolves
            if(resolve==0):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']
                m.tp.dcost = config['trajectory']['tp']['dcost']
                m.phi.dcost = config['trajectory']['phi']['dcost']
                m.options.nodes = 2
            elif(resolve==1):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']*1.3
                m.tp.dcost = config['trajectory']['tp']['dcost']*1.3
                m.phi.dcost = config['trajectory']['phi']['dcost']*1.3
                m.options.nodes = 2
            elif(resolve==2):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']*1.5
                m.tp.dcost = config['trajectory']['tp']['dcost']*1.5
                m.phi.dcost = config['trajectory']['phi']['dcost']*1.5
                m.options.nodes = 3
                
            # Solve the horizon
            m.solve(remote=False)
    
            # Check to see if APMonitor returned successful
            success_check = m.options.appstatus
            if not success_check:
                resolve = resolve + 1

        print('*************************************')
        print('*************************************')
        print('')
        print('End Iteration: ' +str(i))
        print('')
        print('*************************************')
        print('*************************************')
    
    #%% Retrieve results
    
        # Get Solution
        sol = m.load_results()
        solData = pd.DataFrame.from_dict(sol) # convert APM solution to dataFrame
        # If this is the last time step, grab the whole horizon
        if(i==range(0,length,time_shift)[-1]):
            solData = solData.iloc[1:,:]
        # Otherwise we just need the first time_shift number of points
        else:
            solData = solData.iloc[1:time_shift+1,:]
        solData['successful_solution'] = success_check
        solData['iterations'] = 0
        solData['objfcnval'] = m.options.objfcnval
        solData['timestamp'] = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
        solData['iteration_time'] = tm.time()-iter_start
        solData['re-solve'] = resolve
        # Append to full array
        if(config['wind']['use_wind']==False):
            columns = ['time', 'tp', 'phi', 'theta', 'alpha', 'gamma', 'psi', 'v', 'x', 'y', 'h','hdot', 'dist', 'te', 'e_batt', 'p_bat', 'p_n', 'p_solar', 'p_total', 'panel_efficiency',
                   'd', 'c_d', 'c_d_p', 'cl','l', 'rho', 'mu', 't_air', 're', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3', 'sunset', 'mu_clipped', 
                   'mu_slack', 'timestamp', 'iterations', 'objfcnval', 'successful_solution', 
                   'iteration_time', 're-solve','e_balance','workd','worktp','ke','pe']
        else:
            columns = ['time', 'tp', 'phi', 'theta', 'alpha', 'beta', 'gamma', 'psi','chi', 'v_g', 'v_a', 'x', 'y', 'h','x_a','y_a','h_a','hdot', 'dx', 'dy' ,'dist', 'te', 'e_batt', 'p_bat', 'p_n', 'p_solar', 'p_total', 'panel_efficiency',
                   'd', 'c_d', 'c_d_p', 'cl','l', 'rho', 'mu', 't_air', 're', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3','w_n','w_e','w_d', 'sunset', 'mu_clipped', 
                   'mu_slack', 'timestamp', 'iterations', 'objfcnval', 'successful_solution', 
                   'iteration_time', 're-solve','e_balance','workd','worktp','ke','pe']

        # Reorder the columns as desired
        solData_custom = solData[columns]
        # Reindex time to match up to time since dawn
        solData_custom['time'] = solData_custom['t'] - start_time * 3600
        solData_custom.columns = columns
        # Append this to the data history
        dataOut = dataOut.append(solData_custom) 
        
        # Save convex weights
        df_prev = pd.DataFrame.from_dict(sol)
        df_prev = df_prev.shift(-time_shift)
        df_prev = df_prev.fillna(value=0)
        
        # Save data to file at specified frequency
        print('Iteration Time: ' +str(tm.time()-iter_start))
        if(i%save_freq==0):
            ## Results to File
            print('Saving Intermediate Results...')
            
            # Create intermediates folder if needed
            path = config['file']['new_path'] + '/Intermediates' 
            if not os.path.exists(path):
                os.makedirs(path)
                
            filename_out = '/iter_'+str(i).zfill(4)+'.xlsx'      
            dataOut_custom = dataOut[columns]
            dataOut_custom.to_excel(path + filename_out,index=False)
            
            # Clean up old files, only last 5 iterations are saved
            data_file_list = glob.glob(path+'/*.xlsx')
            if(len(data_file_list)>5):
                files_to_remove = data_file_list[0:4]
                for file in files_to_remove:
                    try:
                        os.remove(file)
                    except:
                        pass
                
        # Update status on webpage
        # Requires J drive is mapped
        if(i%config['optimization']['status_update_frequency']==0):
            name = time_stamp
            hour = np.round(solData_custom.time.iloc[-1]/3600.0,2)
            percent = int(np.round((hour/24.0*100)))
            iteration = i/time_shift
            failed = (len(dataOut) - dataOut['successful_solution'].sum() - 1)/time_shift
            last_ten = (len(dataOut['successful_solution'].tail(10*time_shift)) - dataOut['successful_solution'].tail(10*time_shift).sum())/time_shift
            TE = np.round(float(solData.te.iloc[-1]),1)
            runTime = np.round((tm.time()-start)/3600.0,1)
            description = config['file']['description']
            try:
                updateStatus(name,hour,percent,iteration,failed,last_ten,TE,runTime,description)
            except:
                print('J drive not mapped, cannot update status.')
            if(last_ten==10):
                print('!!!!!!!!!! Ten failed solutions in a row.  Shutting down. !!!!!!!!!!!!!')
                sys.exit()
                
#        # Grab infeasibility file if failed
#        if(success_check==0):
#            # Create intermediates folder if needed
#            path = config['file']['new_path'] + '/Infeasibilities' 
#            if not os.path.exists(path):
#                os.makedirs(path)
#                
#            filename_inf = '/infeasibilities' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + '_iteration_'+str(i)+'.txt'      
#    
#            apm_get(server,app,'infeasibilities.txt')
#            
#            try:
#                shutil.move('infeasibilities.txt',path+filename_inf)
#            except:
#                print('Could not retrieve infeasibilities.txt.')    
#            
#            sys.exit()
    
    #%%
    
    ## Results to File
    filename_out = './opt_results_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + '.xlsx' # Saves in current folder
    dataOut_custom.to_excel(filename_out,index=False)
    print('Results retrieved')
    
    print('OPTIMIZATION COMPLETE')
    
    end = tm.time()
    solveTime = end - start
    start = tm.time()
    
    #%%
    print('Plotting...')
    # Plot 3D Path
    plot3DPath(dataOut_custom, 1)
    
    end = tm.time()
    plotTime = end - start
    totalTime = solveTime + plotTime
    print("Solve Time: " + str('{:.2f}'.format(solveTime/3600)) + ' hrs')
    print("Plot Time: " + str(plotTime))
    print("Total Time: " + str(totalTime))