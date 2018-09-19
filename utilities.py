# -*- coding: utf-8 -*-
import os
import pickle
import datetime

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
        
    # Save configuration file to output folder
    config_filepath = os.path.join(results_folder,'config_file_' + str(time_stamp) +'.pkl')
    save_object(config,config_filepath)
        
    print('Successfully created configuration file in ' + results_folder)
        
    return config

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

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