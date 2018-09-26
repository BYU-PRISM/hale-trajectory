# -*- coding: utf-8 -*-

from settings import Settings, process_settings
from utilities import setup_directories
from steady_state import circular_orbit
from define_model import define_model
from optimize import optimize_trajectory
from utilities import save_config
from plotting import plot_all
from solar import SolarLocations

#%% Setup
# Load configuration settings
config = Settings()
config.solar = SolarLocations.albuquerque_summer_solstice
config.description = 'Test SM Summer'

# Process configuration settings
config = process_settings(config)

# Create directory for run results
config = setup_directories(config)

# Save copy of settings in output folder
save_config(config)

#%% Initialization
# Steady state solution
circular_orbit(config)

# Initialize GEKKO model        
m = define_model(config)

#%% Solve
# Optimize
optimize_trajectory(m,config)

#%% Post Processing and Plotting
plot_all(config.results_folder,config)
