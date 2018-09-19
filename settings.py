# -*- coding: utf-8 -*-

import numpy as np
from pydoc import locate
import sys

from utilities import SolarLocation, Param

class Settings:
    
    def __init__(self):
        # Mission Settings
        self.distance = Param(
                    max = 3000 # Maximum flight distance from center point
                    )
        self.x = Param(
                    initial_value = 0, # Initial airplane x position
                    max = self.distance.max,
                    min = -self.distance.max,
                    units = 'm'
                    )
        self.y = Param(
                    initial_value = -self.distance.max, # Initial airplane y position
                    max = self.distance.max,
                    min = -self.distance.max,
                    units = 'm'
                    )
        self.h = Param(
                    initial_value = 18288, # Initial airplane altitude
                    max = 24289,
                    min = 18288,
                    units = 'm'
                    )
        self.initial_direction = 'Clockwise' # CLockwise (default) or Counterclockwise
        
        # Environment
        self.solar = SolarLocations.albuquerque_winter_solstice # Time and location for solar calculations
        self.use_wind = False
        self.w_n = 0 # Wind north component
        self.w_e = 0 # Wind east component
        self.w_d = 0 # Wind down component
        
        # Aircraft settings
        # Aircraft data will be loaded in the process_settings command based on this name
        self.aircraft_name = 'aquila_e216' # Must match filename in Aircraft folder
        
        # Optimization settings
        self.time_step = Param(
                    value = 8,
                    units = 's'
                    )
        self.horizon_length = Param(
                    value = 15,
                    units = 'min'
                    )
        self.time_shift = Param(
                    value = 80,
                    units = 's'
                    )
        self.iteration_save_frequency = 1
        self.max_iterations = 3000
        self.server = 'https://byu.apmonitor.com'
        self.linear_solver = 'ma57'
        
        # Output folder label
        self.description = 'Test'
        self.full_description = 'Insert full description here'

class SolarLocations:
    albuquerque_winter_solstice = SolarLocation(
                latitude = 35.0853,
                longitude = -106.6056,
                elevation = 1.619, # Units?
                altitude = 20, # Probably km
                year = 2016,
                month = 12,
                day = 21,
                zone = -7, # Time zone relative to GMT
                name = 'albuquerque_winter_solstice'
                )
    albuquerque_summer_solstice = SolarLocation(
                latitude = 35.0853,
                longitude = -106.6056,
                elevation = 1.619, # Units?
                altitude = 20, # Probably km
                year = 2016,
                month = 6,
                day = 20,
                zone = -7, # Time zone relative to GMT
                name = 'albuquerque_summer_solstice'
                )        
    albuquerque_spring_equinox = SolarLocation(
                latitude = 35.0853,
                longitude = -106.6056,
                elevation = 1.619, # Units?
                altitude = 20, # Probably km
                year = 2016,
                month = 3,
                day = 19,
                zone = -7, # Time zone relative to GMT
                name = 'albuquerque_spring_equinox'
                )      
    albuquerque_fall_equinox = SolarLocation(
                latitude = 35.0853,
                longitude = -106.6056,
                elevation = 1.619, # Units?
                altitude = 20, # Probably km
                year = 2016,
                month = 9,
                day = 22,
                zone = -7, # Time zone relative to GMT
                name = 'albuquerque_fall_equinox'
                )    
    gabon_winter_solstice = SolarLocation(
                latitude = 0.4162,
                longitude = 9.4673,
                elevation = 0, # Units?
                altitude = 25, # Probably km
                year = 2016,
                month = 12,
                day = 21,
                zone = 1, # Time zone relative to GMT
                name = 'gabon_winter_solstice'
                )

def process_settings(config):
    '''
    Additional pre-calculations and cleanup of settings file
    '''
    
    # Load aircraft settings based on name
    sys.path.insert(0,'./Aircraft/')
    Aircraft = locate('Aircraft.' + config.aircraft_name + '.Aircraft')
    config.aircraft = Aircraft()
    
    # Calculate starting and ending times
    solar_data = config.solar.name
    
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
    config.start_time = Param(
                        value = start_time,
                        units = 'hrs'
                        )
    config.end_time = Param(
                        value = start_time + 24,
                        units = 'hrs'
                        )
        
    # Update initial heading and bank angle based on clockwise/counter clockwise start
    initial_direction = config.initial_direction
    # Configure clockwise or counterclockwise start
    if initial_direction == 'Clockwise': # NOTE - this defintion is true relative to North and East, but the xy coordinates are flipped from this.
        initial_heading = 0
        phi_0 = 0.034
    elif initial_direction == 'Counterclockwise':
        initial_heading = np.pi # 180 degrees
        phi_0 = -0.034
    config.aircraft.psi.initial_value = initial_heading
    config.aircraft.phi.initial_value = phi_0
    
    horizon = config.horizon_length.value
    time_step = config.time_step.value
    distance = config.distance.max
    time_shift = config.time_shift.value
    
    # Convert Horizon from time to timesteps
    config.horizon_steps = int(horizon/time_step*60*distance/3000)
    
    # Convert timeshift from time to timesteps
    config.timeshift_steps = int(np.round(time_shift/time_step))
    
    return config

