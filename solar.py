#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:24:54 2017

@author: kimberlyharris
"""

import numpy as np
import pandas as pd
import os
from os import remove
from numpy import cos,sin,radians
from subprocess import Popen
import subprocess
import shutil
import sys

def loadSmartsData(lat=35.0853,lon=-106.6056,elevation=1.609, altitude=25, year=2016,
              month=12, day=21, zone=-7):
    # Check if file already exists, if not, call SMARTs
   # inputs = list(zip(lat, lon, elevation, altitude, year, month, day, hour, zone, time_interval, phi, theta, psi))
    inputs_string = '{:.4f}'.format(lat) + '_' + '{:.4f}'.format(lon) +'_' + str(altitude) + '_' +str(elevation) + '_' +str(year) + '_' +str(month) + '_' +str(day) + '_'  +str(zone)

    cache_folder = os.path.join(os.getcwd(),'pySMARTS','Cached_Data')
    file = os.path.join(cache_folder,inputs_string + ".csv")
    if os.path.isfile(file):
        smartsData = pd.read_csv(file,delimiter = ',')
    else:
        print("no such file") 
        time_interval = 60 # seconds
        time_range = np.arange(0,3600*24+time_interval,time_interval)
        data1 = pySMARTS(lat,lon, elevation, altitude, year, month, day, zone, time_range)
        C = ['hour', 'Global Horizonal (W/m^2)', 'Global Tracking (W/m^2)', 'Direct Horizontal', 'Direct Tracking', 'Diffuse Horizontal', 'Diffuse Tracking', 'Solar Zenith (deg)', 'Solar Azimuth (deg)', 'Surface Tilt (deg)', 'Surface Azimuth (Deg From North)']
        df = pd.DataFrame(data=data1) 
        df.to_csv(file,  index=False , header=C) 
        smartsData = pd.read_csv(inputs_string+'.csv',delimiter = ',')
    
    return smartsData

def solarFlux(smartsData, hour, phi, theta, psi):
    '''----INPUTS------
    smartsData    Data file from SMARTS - use loadSmartsData
    hour          LST
    phi           Bank angle PITCH                     radians
    theta         Flight path angle (no wind) YAW      radians
    psi           Heading (clockwise from north) ROLL  radians
    '''
    
    # Unpack data from SMARTS
    SMARTS_time = smartsData['hour']
    SMARTS_flux = smartsData['Direct Tracking']
    SMARTS_azimuth = smartsData['Solar Azimuth (deg)']
    SMARTS_zenith = smartsData['Solar Zenith (deg)']
    SMARTS_horizontal_flux = smartsData['Direct Horizontal']

    if(hour>24):
        hour = hour%24 # Wrap back to beginning of day if needed
    time = hour
    
    # Interpolate
    flux = np.interp(time,SMARTS_time,SMARTS_flux)
    azimuth = np.interp(time,SMARTS_time,SMARTS_azimuth)
    zenith = np.interp(time,SMARTS_time,SMARTS_zenith)
    h_flux = np.interp(time,SMARTS_time,SMARTS_horizontal_flux)
        
    # Solar position
    A = radians(-azimuth + 90) # Convert azimuth angle to standard math coordinates (x-axis = 0 , positive counter-clockwise)
    Z = radians(zenith)
    
    # Convert sun angles to vector
    u = cos(A)*sin(Z)
    v = sin(A)*sin(Z)
    w = cos(Z)
    sun_vector = np.array([u,v,w])
    
    # Calculate surface normal in inertial frame
    c1 = cos(-phi)
    c2 = cos(-theta)
    c3 = cos(psi)
    s1 = sin(-phi)
    s2 = sin(-theta)
    s3 = sin(psi) 
   
    # Calculate surface normal in inertial frame
    n1 = c1*s2*s3 - c3*s1
    n2 = c1*c3*s2 + s1*s3
    n3 = c1*c2
    surface = np.array([n1,n2,n3])
    
    # Obliquity factor (0-1)
    mu = np.dot(sun_vector/np.linalg.norm(sun_vector),surface/np.linalg.norm(surface))
    if(mu<0): # Clip to zero
        mu = 0

    # Adjust direct flux from SMARTS for panel orientation
    totalFlux = mu * flux

    solar_data = np.c_[totalFlux, zenith, azimuth, h_flux, mu, u, v, w, flux]
    return solar_data

def pySMARTS(lat,lon,elevation,altitude,year,month,day,zone,time_range):
    
    # Initialize zenith and azimuth
    last_good_sol_zen = 0
    last_good_sol_az = 0
    
    # Loop through the day
    for hr in time_range/3600.0: 
        # Record Time
        hour = hr
    
        # Delete existing input and output files
        if os.path.isfile('pySMARTS/smarts295.inp.txt'):
            remove('pySMARTS/smarts295.inp.txt')
        if os.path.isfile('pySMARTS/smarts295.ext.txt'):
            remove('pySMARTS/smarts295.ext.txt')
        if os.path.isfile('pySMARTS/smarts295.out.txt'):
            remove('pySMARTS/smarts295.out.txt')
                
        # Copy template file
            shutil.copy('pySMARTS/inputTemplate.txt','pySMARTS/smarts295.inp.txt')
    
        # Write new input file
        replacements = {'lat':str(lat), 'lon':str(lon), 'elevation':str(elevation), 'alt':str(altitude), 'year':str(year), 'month':str(month), 'day':str(day), 'hour':str(hour), 'zone':str(zone)}
        # Python 3
        if(sys.version_info > (3, 0)):
            with open('pySMARTS/inputTemplate.txt') as infile, open('pySMARTS/smarts295.inp.txt', 'w') as outfile:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    outfile.write(line)
        # Python 2
        else:
            with open('pySMARTS/inputTemplate.txt') as infile, open('pySMARTS/smarts295.inp.txt', 'w') as outfile:
                for line in infile:
                    for src, target in replacements.iteritems():
                        line = line.replace(src, target)
                    outfile.write(line)
        
        # Run SMARTS with new input
        os.chdir('pySMARTS/')
        print('Hour: ' + str('{0:.2f}'.format(hour)))
        p = Popen("smarts295bat.exe",shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        stdout, stderr = p.communicate(input='\n\n\r'.encode())
        os.chdir('../')
        
        # Parse output file to get global irradiance for horizontal and tracking surfaces
        i = 0
        success = 0
        with open('pySMARTS/smarts295.out.txt', 'rb') as f: # This line has an error w/config file
            for line in f:
                if line.startswith(b'  Direct'):
                    lineList = line.split()
                    if(i==0):
                        horiz_global = float(lineList[9])
                        horiz_direct = float(lineList[3])
                        horiz_diffuse = float(lineList[6])
                        i = i+1
                        success = 1
                    elif(i==1):
                        track_global = float(lineList[14])
                        track_direct = float(lineList[3])
                        track_diffuse = float(lineList[7])
                        i = i+1
                if line.startswith(b'    Zenith'):
                    lineList = line.split()
                    sol_zen = float(lineList[4])
                    sol_az = float(lineList[9])
                    last_good_sol_zen = sol_zen
                    last_good_sol_az = sol_az
                if line.startswith(b'   Surface'):
                    lineList = line.split()
                    surf_tilt = float(lineList[3])
                    surf_az = float(lineList[9])
        if(success==1):
            success=0
        else:
            horiz_global = 0
            track_global = 0
            horiz_direct = 0
            horiz_diffuse = 0
            track_direct = 0
            track_diffuse = 0
            sol_zen = last_good_sol_zen # Repeat the last good zenith and azimuth to keep other things from breaking
            sol_az = last_good_sol_az
            surf_tilt = 0
            surf_az = 0
        data_row = np.c_[hr, horiz_global, track_global, horiz_direct, track_direct, horiz_diffuse, track_diffuse, sol_zen, sol_az, surf_tilt, surf_az]
        if(hr==0):
            data = data_row
        else:
            data = np.r_[data, data_row]

    # Repair bad azimuth and zenith values from morning hours
    data[:(np.nonzero(data[:,7])[0][0]),7] = data[np.nonzero(data[:,7])[0][0],7]
    data[-100:,7] = np.linspace(data[-100,7],data[0,7],num=100)
    data[:(np.nonzero(data[:,8])[0][0]),8] = data[np.nonzero(data[:,8])[0][0],8]
    data[-100:,8] = np.linspace(data[-100,8],data[0,8],num=100)
        
    return data

if __name__ == "__main__":
    smartsData = loadSmartsData()
    solar_data = solarFlux(smartsData)
    print(solar_data)
    
    loadSmartsData(35.000,20.000,18.3,0,2016,12,21,1)
