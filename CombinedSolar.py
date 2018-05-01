#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:24:54 2017

@author: kimberlyharris
"""

import numpy as np
import pandas as pd
import os
from numpy import cos,sin,radians
from runSMARTS import pySMARTS
import sys

def loadSmartsData(lat=35.0853,lon=-106.6056,elevation=1.609, altitude=25, year=2016,
              month=12, day=21, zone=-7):
    # Check if file already exists, if not, call SMARTs
   # inputs = list(zip(lat, lon, elevation, altitude, year, month, day, hour, zone, time_interval, phi, theta, psi))
    inputs_string = '{:.4f}'.format(lat) + '_' + '{:.4f}'.format(lon) +'_' + str(altitude) + '_' +str(elevation) + '_' +str(year) + '_' +str(month) + '_' +str(day) + '_'  +str(zone)

    files = inputs_string + ".csv"
    if os.path.isfile(files):
        smartsData = pd.read_csv(inputs_string+'.csv',delimiter = ',')
    else:
        print("no such file") 
        time_interval = 60 # seconds
        time_range = np.arange(0,3600*24+time_interval,time_interval)
        data1 = pySMARTS(lat,lon, elevation, altitude, year, month, day, zone, time_range)
        C = ['hour', 'Global Horizonal (W/m^2)', 'Global Tracking (W/m^2)', 'Direct Horizontal', 'Direct Tracking', 'Diffuse Horizontal', 'Diffuse Tracking', 'Solar Zenith (deg)', 'Solar Azimuth (deg)', 'Surface Tilt (deg)', 'Surface Azimuth (Deg From North)']
        df = pd.DataFrame(data=data1) 
        df.to_csv(inputs_string+'.csv',  index=False , header=C) 
        smartsData = pd.read_csv(inputs_string+'.csv',delimiter = ',')
    
    return smartsData

def solarFlux(smartsData=None, lat=35.0853,lon=-106.6056,elevation=1.609, altitude=25, year=2016,
              month=12, day=21, hour=11, zone=-7, orientation=False, phi=0, theta=0, psi=0, panel_efficiency=.25, airfoil=False,
              height_correct=False, panel_correct=False):
    '''----INPUTS------
    latitude                                           degrees
    lon                                                degrees
    elevation     elevation above sea level            km
    altitude      height above ground                  km
    year 
    month
    day 
    hour          LST
    zone 
    phi           Bank angle PITCH                     radians
    theta         Flight path angle (no wind) YAW      radians
    psi           Heading (clockwise from north) ROLL  radians
    panel_efficiency
    :param smartsData: '''
    
    # Unpack data from SMARTS
    if(type(smartsData)!=pd.core.frame.DataFrame):
        raise ValueError('Error: solarFlux requires smartsData as an input')
    else:
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
    
    totalFlux = flux
    
#    if airfoil == True:
#        normals = np.genfromtxt('NORMALS.csv', delimiter=',')
#
#        # Left Wing
#        lwne = normals[:,0]
#        lwnn = normals[:,1]
#        lwnu = normals[:,2]
#        
#        # Right Wing
#        rwne = normals[:,3]
#        rwnn = normals[:,4]
#        rwnu = normals[:,5]
#        
#        areas = normals[:,6]
#        
#        totalFlux = 0
#        for i in range(lwne.shape[0]):
#            a11 = c1*c3 + s1*s2*s3
#            a12 = c3*s1*s2 - c1*s3
#            a13 = c2*s1
#            a21 = c2*s3
#            a22 = c2*c3
#            a23 = -s2
#            a31 = c1*s2*s3 - c3*s1
#            a32 = c1*c3*s2 + s1*s3
#            a33 = c1*c2
#            #create rotation matrix
#            A = np.matrix([[a11, a12, a13],
#                           [a21, a22, a23],
#                           [a31, a32, a33]])
#            surface = np.array([lwnn[i],lwne[i],lwnu[i]]*A).ravel()
#             # Obliquity factor (0-1)
#            mu = np.dot(sun_vector/np.linalg.norm(sun_vector),surface/np.linalg.norm(surface))
#
#            # Adjust direct flux from SMARTS
#            P_solar = mu * panel_efficiency * flux * height_factor * areas[i]
#            
#            # Adjust direct flux from SMARTS
#            totalFlux = totalFlux + P_solar
#        for i in range(rwne.shape[0]):
#            surface = np.array([rwnn[i],rwne[i],rwnu[i]]*A).ravel()
#             # Obliquity factor (0-1)
#            mu = np.dot(sun_vector/np.linalg.norm(sun_vector),surface/np.linalg.norm(surface))
#            
#            # Adjust direct flux from SMARTS
#            P_solar = mu * panel_efficiency * flux * height_factor * areas[i]
#
#            # Adjust direct flux from SMARTS
#            totalFlux = totalFlux + P_solar     
            
    if(height_correct==True):
        # Calculates a height factor
        h = altitude
        h = 20000 #meters, altitude above the earth's surface
        height_factor = (-2.82102*10.**(-19.)*(h**4.) + 3.35523*10.**(-14.)*(h**3.) - 1.56318*10.**(-9.)*(h**2.) + 3.42287*10.**(-5.)*h+.698624)
        totalFlux = totalFlux * height_factor
        
    if(orientation==True):     
        # Calculate surface normal in inertial frame
        n1 = c1*s2*s3 - c3*s1
        n2 = c1*c3*s2 + s1*s3
        n3 = c1*c2
        surface = np.array([n1,n2,n3])
        
        # Obliquity factor (0-1)
        mu = np.dot(sun_vector/np.linalg.norm(sun_vector),surface/np.linalg.norm(surface))
        if(mu<0):
            mu = 0

        # Adjust direct flux from SMARTS
        totalFlux = mu * flux
    else:
        mu = 0
        
    if(panel_correct==True):
        totalFlux = panel_efficiency * flux
    
    solar_data = np.c_[totalFlux, zenith, azimuth, h_flux, mu, u, v, w, flux]
    return solar_data

if __name__ == "__main__":
    smartsData = loadSmartsData()
    solar_data = solarFlux(smartsData)
    print(solar_data)
    
    loadSmartsData(35.000,20.000,18.3,0,2016,12,21,1)
