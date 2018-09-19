# -*- coding: utf-8 -*-
import os
import yaml
import datetime

def setup_directories(config):
    '''
    Creates a new time-stamped directory for the new optimization results.
    '''
    
    # Get current time
    time_stamp = str('{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    # Create new results folder from timestamp and description
    cwd = os.getcwd()
    results_folder = os.path.join(cwd,'Results','hale_' + str(time_stamp) + ' - ' + config['file']['description'])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    # Add new results folder to config
    config['file']['results_folder'] = results_folder
    config['file']['time_stamp'] = str(time_stamp)
        
    # Save configuration file to output folder
    config_filepath = os.path.join(results_folder,'config_file_' + str(time_stamp) +'.yml')
    config['file']['config_filepath'] = config_filepath
    with open(config_filepath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
    print('Successfully created configuration file in ' + results_folder)
        
    return config