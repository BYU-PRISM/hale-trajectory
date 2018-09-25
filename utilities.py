# -*- coding: utf-8 -*-
import os
import pickle
import jsonpickle
import datetime
import copy

def setup_directories(config):
    '''
    Creates a new time-stamped directory for the new optimization results.
    '''
    
    # Get current time
    time_stamp = str('{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    # Create new results folder from timestamp and description
    cwd = os.getcwd()
    results_folder = os.path.join(cwd,'Results','hale_' + str(time_stamp) + ' - ' + config.description)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    # Add new results folder to config
    config.results_folder = results_folder
    config.time_stamp = str(time_stamp)

    print('Successfully created output folder ' + results_folder)
        
    return config

def save_config(config):
    # Save configuration file to output folder
    results_folder = config.results_folder
    time_stamp = config.time_stamp
    config_filepath = os.path.join(results_folder,'config_file_' + str(time_stamp) +'.pkl')
    save_pickle(config,config_filepath)
    config.config_filepath_pickle = config_filepath
    config_filepath = os.path.join(results_folder,'config_file_' + str(time_stamp) +'.json')
    save_json(config,config_filepath)
    config.config_filepath_json = config_filepath
    
    print('Successfully created configuration file in ' + results_folder)
    

def save_pickle(config, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(config, output, pickle.HIGHEST_PROTOCOL)
        
def save_json(config, filename):
    obj = copy.deepcopy(config)
    del obj.solar.smartsData # Remove smartsData before saving json
    with open(filename, 'w') as output:  # Overwrites any existing file.
        output.write(jsonpickle.encode(obj,unpicklable=False))
        
def load_pickle(filename):
    obj = pickle.load(open(filename,'rb'))
    return obj

def init_state():
    '''
    Helper function for state machine
    '''
    global state
    state = 0

class SolarLocation:
    def __init__(self,latitude,longitude,elevation,altitude,year,month,day,zone,name):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation # km
        self.altitude = altitude # km
        self.year = year
        self.month = month
        self.day = day
        self.zone = zone
        self.name = name
        self.smartsData = None # Will be loaded later
        
class Param:
    '''
    Generic parameter holder
    '''
    def __init__(self,value=None,units=None,initial_value=None,max=None,min=None):
        self.initial_value = initial_value
        self.value = value
        self.max = max
        self.min = min
        self.units = units
        
class Var:
    '''
    Generic variable holder
    '''
    def __init__(self,ss_initial_guess=None,max=None,min=None,dmax=None,
                 dcost=None,units=None,description=None,up=None,down=None,
                 level=None,mode=None):
        self.ss_initial_guess = ss_initial_guess
        self.max = max
        self.min = min
        self.dmax = dmax
        self.dcost = dcost
        self.units = units
        self.description = description
        self.mode = mode
        self.up = up
        self.down = down
        self.level = level