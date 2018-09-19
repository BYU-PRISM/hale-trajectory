# -*- coding: utf-8 -*-

import numpy as np

def load_settings():
    '''
    Defines the settings used for the optimization
    '''
    
    print("Defining Configuration File")
    
    time_step = 8 # 15 # seconds int(30/n)
    distance = 3000
    horizon = 15 # minutes (12.5 default)
    horizon_length = int(horizon/time_step*60*distance/3000) # Number of timesteps
    time_shift_time = 48*time_step # seconds !!! Must be a multiple of the timestep (rounds otherwise)
    time_shift = int(np.round(time_shift_time/time_step)) #10 # Number of timesteps to shift forward
    battery_mass = 136.7#113#141.1#153.9#141.1#142.5#139.5
    total_mass = 213 + battery_mass #355.5#352.5
    max_iterations = 10000
    initial_direction = 'Clockwise' # Options: 'Clockwise', 'Counterclockwise'
    server = 'https://byu.apmonitor.com' # 'https://xps.apmonitor.com' # 'http://127.0.0.1' # Other options
    linear_solver = 'ma57' # Other options (only uses this in certain cases, depending on model file chosen)
    h_0 = 18288
    alpha_dmax = 6.5*np.pi/180.0/(30.0/time_step)
    alpha_dcost = 0.5/(30.0/time_step)*0.7*15
    tp_dcost = 0.05/(30.0/time_step)/100*15#0.05/(30.0/time_step)/100*15
    phi_dcost = 0.5/(30.0/time_step)*15
    wind = [0,10,0]
    define_use_wind = False
    gamma_factor = 2
    name = 'Aquila E216 New'
    description = 'Test'
    
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
        phi_0 = 0.034
    elif initial_direction == 'Counterclockwise':
        initial_heading = np.pi # 180 degrees
        phi_0 = -0.034
        
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
                max = 24289,#27432, # 25000, ~82,000 ft
                initial_value = h_0,
                units = 'm',
                ),
            initial_direction = initial_direction, # CLockwise (default) or Counterclockwise
            distance_from_center = dict(
                initial_value = 0,
                units = 'm',
                ),
            v = dict(
                ss_initial_guess = 33,
                initial_value = 33,
                units = 'm/s',
                ),
            gamma = dict(
                max = float(np.radians(5)),
                min = float(np.radians(-5)),
                initial_value = 0,
                up = 0.04/gamma_factor,
                level = 0,
                down = -0.024/gamma_factor,
                mode = 'level',
                units = 'radians',
                description = 'flight path angle (pitch)',
                ),
            psi = dict(
                initial_value = 0, # This will be updated in process_settings()
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
                dmax = 50,
                dcost = tp_dcost,
                ss_initial_guess = 75,
                initial_value = 0, # Opt initial value for first timestep. This will be updated after the SS code is run
                units = 'Newtons',
                description = 'Thrust',
                ),
            alpha = dict(
                max = float(np.radians(15)),
                min = float(np.radians(-10)),
                dmax = alpha_dmax,
                dcost = alpha_dcost,
                ss_initial_guess = 0.069,
                initial_value = 0, # Opt initial value for first timestep. This will be updated after the SS code is run            
                units = 'radians',
                description = 'angle of attack',
                ),
            phi = dict(
                max = float(np.radians(5)),
                min = float(np.radians(-5)),
                dmax = 0.08/(30.0/time_step),
                dcost = phi_dcost,
                cost = 0,
                ss_initial_guess = 0.034, # This will be updated in process_settings() to adjust for direction
                initial_value = 0, # Opt initial value for first timestep. This will be updated after the SS code is run                        
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
                value = 0, # This will be updated in process_settings()
                units = 'hr',
                ),
            end_time = dict(
                value = 0, # This will be updated in process_settings()
                units = 'hr',
                ),
            horizon_length = horizon_length,
            iteration_save_frequency = 1,
            status_update_frequency = 1,
            max_iterations = max_iterations,
            server = server,
            linear_solver = linear_solver,
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
            ),
    )
        
    return config

def process_settings(config):
    '''
    Additional pre-calculations and cleanup of settings file
    '''
    
    # Calculate starting and ending times
    solar_data = config['solar']['solar_data']
    
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
    config['optimization']['start_time']['value'] = start_time
    config['optimization']['end_time']['value'] = start_time + 24
        
    # Update initial heading and bank angle based on clockwise/counter clockwise start
    initial_direction = config['trajectory']['initial_direction']
    # Configure clockwise or counterclockwise start
    if initial_direction == 'Clockwise': # NOTE - this defintion is true relative to North and East, but the xy coordinates are flipped from this.
        initial_heading = 0
        phi_0 = 0.034
    elif initial_direction == 'Counterclockwise':
        initial_heading = np.pi # 180 degrees
        phi_0 = -0.034
    config['trajectory']['psi']['initial_value'] = initial_heading
    config['trajectory']['phi']['ss_initial_guess'] = phi_0
    
    return config