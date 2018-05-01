# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:11:52 2017

Create yaml file for configuration 

Updates:
    7/12/17 - Added iteration_save_frequency and configured in optimize_MPC_config.py
    7/18/17 - Added max_iterations and configured in optimize_MPC_config.py
    7/19/17 - Configured to automatically run whole optimization script from this file
    8/19/17 - Fixed bux where solar data was defined as 'gabon_winter_solstice' always within the config file
    9/15/17 - Changed initial heading to be pi instead of 0
    9/20/17 - Added in clockwise vs counterclockwise initial direction - still need to test this

@author: NathanielGates
"""

#%% Define configuration file for trajectory optimization
# When this file is run, it will create a new folder and store the configuration file inside it.

import yaml
import datetime
import os
import numpy as np
from optimize_MPC_config import optimize_MPC
from init_model import init_model
#from init_model_wind_level import init_model_wind_level
from integrateSS_wrapper_wind import integrateSS_wrapper_wind

cwd = os.getcwd()

time_stamp = str('{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))

run_whole_optimization = 1 ############### 1 = Yes, 0 = No

if run_whole_optimization == 1:
    option = '2' # Use custom values
else:
    option = str(input('Use Default or Custom Values? \n  1 - Default \n  2 - Custom\n  [1 or 2] : '))

# check if option is equal to one of the strings, specified in the list
if option=='1' : # Default values
    distance = 3000 # Default
    horizon_length = 30
    battery_mass = 212
    total_mass = 425
    time_step = 60
    max_iterations = 2000 # Updated
    initial_direction = 'Clockwise'
    server = 'https://byu.apmonitor.com' # Other options: byu, xps
    linear_solver = 'ma57' # Other options
    h_0 = 18288
    alpha_dmax = 1*np.pi/180.0/(30.0/time_step)
    alpha_dcost = 0.5/(30.0/time_step)
    tp_dcost = 0.05/(30.0/time_step)
    define_use_wind = True
    description = ''
elif option=='2': # Custom values
    n = 3
    distance = 3000
    horizon_length = int(25*n*distance/3000) # 10 minutes
    battery_mass = 140#142.5#139.5
    total_mass = 213+battery_mass#355.5#352.5
    time_step = int(30/n) # seconds
    max_iterations = 3000
    initial_direction = 'Clockwise' # Options: 'Clockwise', 'Counterclockwise'
    server = 'https://byu.apmonitor.com' # 'https://xps.apmonitor.com' # 'http://127.0.0.1' # Other options
#    server = 'https://xps.apmonitor.com'
    linear_solver = 'ma57' # Other options (only uses this in certain cases, depending on model file chosen)
    h_0 = 18288
    alpha_dmax = 6.5*np.pi/180.0/(30.0/time_step)
    alpha_dcost = 0.5/(30.0/time_step)*0.7*15
    tp_dcost = 0.05/(30.0/time_step)/100*15
    phi_dcost = 0.5/(30.0/time_step)*15
    wind = [0,10,0]
    define_use_wind = True
    time_shift = 10
    gamma_factor = 1
    name = 'Aquila'
    description = 'Test Wind'

else :
    print('Error: Please input 1 or 2.')

# Display configurable parameter values
print()
print("Defining Configuration File")
print()
print("distance =",distance)
print("horizon_length =",horizon_length)
print("battery_mass =",battery_mass)
print("total_mass =",total_mass)
print("time_step =",time_step)
print()

# Choose solar data
solar_data = 'albuquerque_winter_solstice' 
#solar_data = 'albuquerque_spring_equinox'
#solar_data = 'albuquerque_summer_solstice'
#solar_data = 'albuquerque_fall_equinox'
#solar_data = 'gabon_winter_solstice'




# Assuming a Dawn start time
if solar_data == 'gabon_winter_solstice':
    start_time = 22800.0/3600
elif solar_data == 'albuquerque_winter_solstice':
    start_time = 26110.0/3600
elif solar_data == 'albuquerque_summer_solstice':
    start_time = 17830.0/3600
elif solar_data == 'albuquerque_spring_equinox':
    start_time = 22440.0/3600
elif solar_data == 'albuquerque_fall_equinox':
    start_time = 21490.0/3600
else:
    print('ERROR - Need to define solar data correctly.')

# Configure clockwise or counterclockwise start
if initial_direction == 'Clockwise': # NOTE - this defintion is true relative to North and East, but the xy coordinates are flipped from this.
    initial_heading = 0
    phi_0 = 0.0656
elif initial_direction == 'Counterclockwise':
    initial_heading = np.pi # 180 degrees
    phi_0 = -0.0656

if (server == 'localhost' or server == 'http://127.0.0.1'):
    if(define_use_wind==True):
        model_file = 'model_template_density_climb_drag_opt_wind_beta.apm'
    else:
        model_file = 'model_template_density_climb_drag_opt_drag_solver_clean_dragsurface.apm'        
#        model_file = 'model_template_density_climb_drag_opt_drag.apm'
#        model_file = 'model_template_density_climb_drag_opt.apm'
    define_linear_solver = 'DEFAULT'
else:
    if(define_use_wind==True):
#        model_file = 'model_template_density_climb_drag_opt_wind_solver.apm'
        model_file = 'model_template_density_climb_drag_opt_wind_solver_clean_beta.apm'
    else:
        model_file = 'model_template_density_climb_drag_opt_drag_solver_clean_dragsurface.apm'
    define_linear_solver = linear_solver # Other options


# Define configuration file
config = dict(
    trajectory = dict(
        x = dict(
            max = distance, # 3000
            min = -distance, # -3000
            initial_value = 0,
            units = 'm',
            ),
        y = dict(
            max = distance, # 3000
            min = -distance, # -3000
            initial_value = -distance, # -3000 # Changed to 3000 (positive) <-- Undid this change
            units = 'm',        
            ),
        h = dict(
            min = 18288, # 60,000 ft
            max = 27432, # 25000, ~82,000 ft
            initial_value = h_0,
            units = 'm',
            ),
        initial_direction = initial_direction, # CLockwise (default) or Counterclockwise
        distance_from_center = dict(
            initial_value = 0,
            units = 'm',
            ),
        v = dict(
            ss_initial_guess = 30,
            initial_value = 30,
            units = 'm/s',
            ),
        gamma = dict(
            max = float(np.radians(5)),
            min = float(np.radians(-5)),
            initial_value = 0,
            up = 0.030/gamma_factor,#0.033/gamma_factor,
            level = 0,
            down = -.018/gamma_factor,#-0.026/gamma_factor,
            mode = 'level',
            units = 'radians',
            description = 'flight path angle (pitch)',
            ),
        psi = dict(
            initial_value = initial_heading, # 0 - changed to pi to face left (clockwise) instead of right (counter-clockwise)
            units = 'radians',
            description = 'heading angle (yaw)',
            ),
        lift_coefficient = dict(
            max = 1.5,
            ),
        # MVs
        tp = dict(
            max = 500,
            min = 0.01,
            dmax = 25/(30.0/time_step),
            dcost = tp_dcost,
            ss_initial_guess = 110.54,
            initial_value = 110.54, # Opt initial value for first timestep. This will be updated after the SS code is run
            units = 'Newtons',
            description = 'Thrust',
            ),
        alpha = dict(
            max = float(np.radians(9)),
            min = float(np.radians(-2)),
            dmax = alpha_dmax,
            dcost = alpha_dcost,
            ss_initial_guess = 0.0874,
            initial_value = 0.0874, # Opt initial value for first timestep. This will be updated after the SS code is run            
            units = 'radians',
            description = 'angle of attack',
            ),
        phi = dict(
            max = float(np.radians(5)),
            min = float(np.radians(-5)),
            dmax = 0.08/(30.0/time_step),
            dcost = phi_dcost,
            cost = 0,
            ss_initial_guess = phi_0, # Need to change this sometime (to be accurate, and used)
            initial_value = phi_0, # Opt initial value for first timestep. This will be updated after the SS code is run                        
            units = 'degrees',
            description = 'bank angle (roll)',
            ),
        ),
    aircraft = dict(
        battery = dict(
            mass = dict(
                value = battery_mass, # 212
                units = 'kg'
                ),  
            energy_density = dict(
                value = 350,
                units = 'Whr/kg',
                ),
            initial_state_of_charge = 0.20, # 0.20,
            ),
        mass_payload = dict(
            value = 25,
            units = 'kg',      
            ),
        mass_total = dict(
            value = total_mass, # 425
            units = 'kg'
            ),
        aspect_ratio = 30,
        wing_top_surface_area = dict(
            value = 60,
            units = 'm^2',
            ),
        propeller_radius = dict(
            value = 2,
            units = 'm',
            ),
        motor_efficiency = 0.95,
        power_for_payload = dict(
            value = 250,
            units = 'W',
            ),
        power_for_internal_systems = dict(
            value = 250,
            units = 'W',
            ),
        airfoil_thickness_ratio = 0.11,
        roughness_factor = 1.1,
        inviscid_span_efficiency = 0.98,
        xcrit_top = 0.7412,
        xcrit_bottom = 1,
        ck = 1.1,
        name = name,
        ),
    optimization = dict(
        time_step = dict(
            value = time_step,
            units = 's',
            ),
        start_time = dict(
            value = start_time, # For Albuquerque = 26160.0/3600, Gabon = 22800.0/3600
            units = 'hr',
            ),
        end_time = dict(
            value = start_time + 24,
            units = 'hr',
            ),
        horizon_length = horizon_length,
        iteration_save_frequency = 1,
        status_update_frequency = 1,
        max_iterations = max_iterations,
        server = server,
        linear_solver = define_linear_solver,
        time_shift = time_shift,
        ),
    solar = dict(
        panel_efficiency = 0.25, # Need to update this
        panel_efficiency_function = dict(
                eta = 0.12,
                beta = 0.0021888986107182,
                Tref = 298.15,
                gamma_s = 0.413220518404272,
                T_noct = 20.0310337470507,
                G_noct = 0.519455027587048,
            ),
        albuquerque_winter_solstice = dict(
            latitude = 35.0853,
            longitude = -106.6056,
            elevation = 1.619, # Units?
            altitude = 20, # Probably km
            year = 2016,
            month = 12,
            day = 21,
            zone = -7, # Time zone relative to GMT
            ),
        albuquerque_summer_solstice = dict(
            latitude = 35.0853,
            longitude = -106.6056,
            elevation = 1.619, # Units?
            altitude = 20, # Probably km
            year = 2016,
            month = 6,
            day = 20,
            zone = -7, # Time zone relative to GMT
            ),          
        albuquerque_spring_equinox = dict(
            latitude = 35.0853,
            longitude = -106.6056,
            elevation = 1.619, # Units?
            altitude = 20, # Probably km
            year = 2016,
            month = 3,
            day = 19,
            zone = -7, # Time zone relative to GMT
            ),       
        albuquerque_fall_equinox = dict(
            latitude = 35.0853,
            longitude = -106.6056,
            elevation = 1.619, # Units?
            altitude = 20, # Probably km
            year = 2016,
            month = 9,
            day = 22,
            zone = -7, # Time zone relative to GMT
            ),       
        gabon_winter_solstice = dict(
            latitude = 0.4162,
            longitude = 9.4673,
            elevation = 0, # Units?
            altitude = 25, # Probably km
            year = 2016,
            month = 12,
            day = 21,
            zone = 1, # Time zone relative to GMT
            ),
        solar_data = solar_data,
        ),
    wind = dict(
        use_wind = define_use_wind, # Use wind?
        w_n = wind[0], # Wind north component
        w_e = wind[1], # Wind east component
        w_d = wind[2], # Wind down component
        ),
    file = dict(
        description = description,
        description_full = 'Insert full description here',
        time_stamp = str(time_stamp),
        folder_name = 'hale_' + str(time_stamp) + ' - ' + description,
        configuration = 'config_file_' + str(time_stamp) +'.yml',
        original_path = cwd,
        new_path = cwd + '/Data/' + 'hale_' + str(time_stamp) + ' - ' + description, # Change to not be computer specific
        model_file_template = model_file,
        ),
)


## Change default values
#battery_mass = 212 - 40 # kg
#config['aircraft']['battery']['mass'] = battery_mass
#config['aircraft']['mass_total'] += battery_mass - 212



# Create new folder
#newpath = cwd + '/Data/' + str(config['file']['folder_name'])
newpath = config['file']['new_path']
if not os.path.exists(newpath):
    os.makedirs(newpath)

os.chdir(newpath)

# Save configuration file
#with open('config_file_' + str(time_stamp) +'.yml', 'w') as outfile:
with open(config['file']['configuration'], 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False) # What is default flow style?

oldpath = config['file']['original_path']
os.chdir(oldpath)

print('Successfully created configuration file in hale_' + str(time_stamp))


#%% Run entire optimization script

if run_whole_optimization == 1:
    
    # Solve steady state
    if(define_use_wind==True):
#        with open('integrateSS_wrapper_wind.py') as source_file:
#            exec(source_file.read())
        config = integrateSS_wrapper_wind(config)
    else:
        with open('integrateSS_wrapper.py') as source_file:
            exec(source_file.read())
    
    # Write model        
    m = init_model(config)
    # Optimize
    optimize_MPC(m,config)
#    from line_profiler import LineProfiler
#    lp = LineProfiler()
#    lp_wrapper = lp(integrateSS_wrapper_wind)
#    lp_wrapper(config)
#    lp.print_stats()
                

#%% Import yaml file and access data from it # Solve steady state
#    if(define_use_wind==True):
#        with open('integrateSS_wrapper_wind.py') as source_file:
#            exec(source_file.read())
#    else:
#        with open('integrateSS_wrapper.py') as source_file:
#            exec(source_file.read())
#    
#    # Write model        
#    m = init_model(config)
#    # Optimize
#    optimize_MPC(m,config)

#ifile = open('config_file.yml', 'r')
#config = yaml.load(ifile)
#
#config['trajectory']['x']['min']
#config['trajectory']['x']['units']
#
#def test(data):
#    x = data['trajectory']['x']['initial_value']
#    y = data['trajectory']['y']['initial_value']
#    h = data['trajectory']['h']['min']
#    add = x + y + h
#    return add
#
#test(config)


#%% How to define a yaml file

#import yaml
#
#data = dict(
#    A = 'a',
#    B = dict(
#        C = 'c',
#        D = 'd',
#        E = 'e',
#    )
#)
#
#with open('data.yml', 'w') as outfile:
#    yaml.dump(data, outfile, default_flow_style=False)
    
    
#%% How to load a yaml file

#ifile = open("data.yml", 'r')
#yfile = yaml.load(ifile)
#ifile.close()
#
#someVar = yfile["A"]        # yaml line looks like:  someVar: 8314.46   # J/kmol*K
## OR
#subVar = yfile["B"]["C"]
