# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time as tm
import datetime
import os
import glob
import sys

def optimize_trajectory(m,config):
    
    # Global Options
    m.options.max_iter = config.max_iterations
    m.options.cv_type = 1
    m.options.time_shift = config.time_shift_steps
    time_shift = config.time_shift_steps
    m.options.csv_read = 2
    m.options.nodes = 2
    m.options.reduce = 4
    m.options.web = 0
    m.options.ctrl_units = 1
    m.options.solver = 3
    m.options.imode = 6
    m.options.otol = 1e-4
    m.options.rtol = 1e-4
    # This was added after the paper
    m.solver_options = ['linear_solver '+ config.linear_solver]
    
    # Setup Variables
    m.alpha.status = 1
    m.alpha.fstatus = 0
    m.alpha.lower = config.aircraft.alpha.min
    m.alpha.upper = config.aircraft.alpha.max
    m.alpha.dmax = config.aircraft.alpha.dmax
    m.alpha.dcost = config.aircraft.alpha.dcost
    m.phi.status = 1
    m.phi.fstatus = 0
    m.phi.lower = config.aircraft.phi.min
    m.phi.upper = config.aircraft.phi.max
    m.phi.dmax = config.aircraft.phi.dmax
    m.phi.dcost = config.aircraft.phi.dcost
    m.tp.status = 1
    m.tp.fstatus = 0
    m.tp.lower = config.aircraft.tp.min
    m.tp.upper = config.aircraft.tp.max
    m.tp.dmax = config.aircraft.tp.dmax
    m.tp.dcost = config.aircraft.tp.dcost
    m.p_bat.status = 1
    m.p_bat.fstatus = 0
    
    #CVs
    m.dist.status = 1
    m.dist.sphi = config.x.max
    m.dist.splo = 0
    m.dist.wsphi = 10
    m.dist.wsplo = 0
    m.dist.tr_init = 0
    
    # Initialize storage from first row of steady state solution
    filenameSim = os.path.join(config.results_folder,'ss_results_' + str(config.time_stamp) + '.xlsx')
    ss_data = pd.read_excel(filenameSim)
    dataOut = ss_data.head(1)
    
    # Load loop parameters
    horizon_length = config.horizon_steps
    time_step = config.time_step.value
    start_time = config.start_time.value
    end_time = config.end_time.value
    
    length = int(3600*(end_time - start_time)/time_step - horizon_length)
    
    save_freq = config.iteration_save_frequency
    
    # Begin timing
    start = tm.time()
    
    # Set horizon
    m.time = ss_data['time'].iloc[0:horizon_length+1].values
    
    # MPC Loop
    for i in range(0,length,time_shift):
        
        # Initialize re-solve to zero, we'll attempt to solve until we succeed or resolve
        # hits the max resolves, adjusting the options each time if needed.
        resolve = 0
        max_resolve = 3
        success_check = 0
        
        while(success_check==0 and resolve < max_resolve):
        
            print('*************************************')
            print('*************************************')
            print('')
            print('Begin Iteration: ' +str(i))
            print('')
            print('*************************************')
            print('*************************************')
        
        
            iter_start = tm.time()
        
            # Constrict MVs at start, this forces the optimizer to stay closer to the 
            # steady state solution for the first few iterations.  Seems to help convergence
            # sometimes
            if(i<250):
                m.tp.dcost = config.aircraft.tp.dcost*(1/((i+1)/250))
                m.phi.dcost = config.aircraft.phi.dcost*(1/((i+1)/250))
                m.alpha.dcost = config.aircraft.alpha.dcost*(1/((i+1)/250))
            else:
                m.tp.dcost = config.aircraft.tp.dcost
                m.phi.dcost = config.aircraft.phi.dcost
                m.alpha.dcost = config.aircraft.alpha.dcost
        
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
                if(config.use_wind):
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
            else:
                # After the first iteration just load new solar values
                df1 = ss_data.iloc[i:i+horizon_length+1,:]
                m.flux.value = df1['flux'].values
                m.t.value = df1['t'].values
                m.zenith.value = df1['zenith'].values
                m.azimuth.value = df1['azimuth'].values
                m.sn1.value = df1['sn1'].values
                m.sn2.value = df1['sn2'].values
                m.sn3.value = df1['sn3'].values
                
            # Adjust settings for progressive re-solves
            if(resolve==0 and i > 250):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']
                m.rpm.dcost = config['trajectory']['rpm']['dcost']
                m.phi.dcost = config['trajectory']['phi']['dcost']
                m.options.nodes = 2
            elif(resolve==1):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']*1.3
                m.rpm.dcost = config['trajectory']['rpm']['dcost']*1.3
                m.phi.dcost = config['trajectory']['phi']['dcost']*1.3
                m.options.nodes = 2
            elif(resolve==2):
                m.alpha.dcost = config['trajectory']['alpha']['dcost']*1.5
                m.rpm.dcost = config['trajectory']['rpm']['dcost']*1.5
                m.phi.dcost = config['trajectory']['phi']['dcost']*1.5
                m.options.nodes = 3
            
            # Optimize over the solution horizon
            m.solve()
            
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
        if(config.use_wind==False):
            columns = ['time', 'tp', 'phi', 'theta', 'alpha', 'gamma', 'psi', 'v', 'x', 'y', 'h', 'dist', 'te', 'e_batt', 'p_bat', 'p_n', 'p_solar', 'p_total', 'panel_efficiency',
                   'd', 'c_d', 'cl', 'rho', 'mu', 't_air', 're', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3', 'sunset', 'mu_clipped', 
                   'mu_slack', 'timestamp', 'iterations', 'objfcnval', 'successful_solution', 
                   'iteration_time', 're-solve']
        else:
            columns = ['time', 'tp', 'phi', 'theta', 'alpha', 'beta', 'gamma', 'psi','chi', 'v_g', 'v_a', 'x', 'y', 'h', 'dx', 'dy' ,'dist', 'te', 'e_batt', 'p_bat', 'p_n', 'p_solar', 'ptotal', 'panel_efficiency',
                   'd', 'cd', 'cl', 'rho', 'mu', 'tair', 're', 'nh', 'nv', 'nu_prop', 't', 'flux', 'g_sol', 'mu_solar', 'azimuth', 'zenith', 'sn1', 'sn2', 'sn3','w_n','w_e','w_d', 'sunset', 'mu_clipped', 
                   'mu_slack', 'timestamp', 'iterations', 'objfcnval', 'successful_solution', 
                   'iteration_time', 're-solve']

        # Reorder the columns as desired
        solData_custom = solData[columns]
        
        # Reindex time to match up to time since dawn
        solData_custom['time'] = solData_custom['t'] - start_time * 3600
        solData_custom.columns = columns
        # Append this to the data history
        dataOut = dataOut.append(solData_custom)
        
        # Save data to file at specified frequency
        print('Iteration Time: ' +str(tm.time()-iter_start))
        if(i%save_freq==0): #
            ## Results to File
            print('Saving Intermediate Results...')
            
            # Create intermediates folder if needed
            path = os.path.join(config.results_folder , 'Intermediates' )
            if not os.path.exists(path):
                os.makedirs(path)
                
            filename_out = '/iter_'+str(i).zfill(4)+'.xlsx'      

            dataOut_custom = dataOut[columns]
            dataOut_custom.to_excel(path + filename_out,index=False)
            
            # Clean up old files
            data_file_list = glob.glob(path+'/*.xlsx')
            if(len(data_file_list)>3):
                files_to_remove = data_file_list[0:2]
                for file in files_to_remove:
                    try:
                        os.remove(file)
                    except:
                        pass
        
        # Check for repeated failed solutions
        last_ten = (len(dataOut['successful_solution'].tail(10*time_shift)) - dataOut['successful_solution'].tail(10*time_shift).sum())/time_shift
        if(last_ten==10):
                print('!!!!!!!!!! Ten failed solutions in a row.  Shutting down. !!!!!!!!!!!!!')
                sys.exit()
        
    
    #%%
    
    ## Save Final Results to File
    filename_out = os.path.join(config.results_folder,'./opt_results_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + '.xlsx')
    dataOut_custom.to_excel(filename_out,index=False)
    print('Results retrieved')
    print('OPTIMIZATION COMPLETE')
    
    end = tm.time()
    solveTime = end - start
    print("Solve Time: " + str('{:.2f}'.format(solveTime/3600)) + ' hrs')