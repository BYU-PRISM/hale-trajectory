# -*- coding: utf-8 -*-

from settings import Settings, process_settings
from utilities import setup_directories
from steady_state import integrate_steady_state
#from optimize import optimize_trajectory
#from init_model import init_model

#%% Setup
# Load configuration settings
config = Settings()

# Process configuration settings
config = process_settings(config)

# Create directories for run results
config = setup_directories(config)

#%% Initialization
# Steady state solution
integrate_steady_state(config)

## Initialize GEKKO model        
#m = init_model(config)
#
##%% Solve
## Optimize
#optimize_trajectory(m,config)
#
##%% Post Processing and Plotting