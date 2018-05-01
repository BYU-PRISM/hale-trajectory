# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:24:21 2017

@author: Jim Guymon
"""

from __future__ import division
import numpy as np
from findSteadyState_wrapper import findSteadyState
from dynamicsWrapper import uavDynamicsWrapper
from scipy.integrate import odeint
import time as tm
from CombinedSolar import solarFlux, loadSmartsData
from plotting import plot3DPath
import datetime
import yaml
import os
mode='automatic'
if mode=='automatic':
    
    ### Import configuration file
    # Run this code after defining the config file (and editing it if needed), with 
    # the variables still in memory.
    
    cwd = os.getcwd()
    newpath = config['file']['new_path']
    oldpath = config['file']['original_path']
    os.chdir(newpath)
    with open(config['file']['configuration'], 'r') as ifile:
        config_file = yaml.load(ifile)
    config = config_file
    os.chdir(oldpath)
    time_stamp = config['file']['time_stamp']
#Setting up guess values and solar data
x0 = [config['trajectory']['tp']['ss_initial_guess'],
      config['trajectory']['alpha']['ss_initial_guess'],
      abs(config['trajectory']['phi']['ss_initial_guess'])] # Note: there is an absolute value here.
solar_data = config['solar']['solar_data'] # <---- Still need to finish debugging this method
#    # Albuquerque NM - Winter Solstice
lat = config['solar'][solar_data]['latitude'] # 35.0853
lon = config['solar'][solar_data]['longitude'] # -106.6056
elevation = config['solar'][solar_data]['elevation'] # 1.619
altitude = config['solar'][solar_data]['altitude'] # 20
year = config['solar'][solar_data]['year'] # 2016
month = config['solar'][solar_data]['month'] # 12
day = config['solar'][solar_data]['day'] # 21
zone = config['solar'][solar_data]['zone'] # -7
smartsData = loadSmartsData(lat,lon,elevation, altitude, year,
              month, day, zone)
      
h=np.linspace(18288,24000,10)
vmin=np.zeros(len(h))
Tpmin=np.zeros(len(h))
alphamin=np.zeros(len(h))
phimin=np.zeros(len(h))
clmin=np.zeros(len(h))
pmin=np.zeros(len(h))
sol=np.zeros((len(h),7))
for i in range(len(h)):
    vmin[i],Tpmin[i],alphamin[i],phimin[i],clmin[i],pmin[i] = findSteadyState(x0,h[i],smartsData,config_file)
    sol[i,0]=h[i]
    sol[i,1]= vmin[i]
    sol[i,2]= Tpmin[i]
    sol[i,3]= alphamin[i]
    sol[i,4]= phimin[i]
    sol[i,5]= clmin[i]
    sol[i,6]= pmin[i]
   
import pandas as pd
solData = pd.DataFrame(sol,columns=('height','vmin','Tpmin','alphamin','phimin','clmin','pmin'))
#dataout=solData[['height','vmin','Tpmin','alphamin','phimin','clmin','pmin']]
filename='heights.xlsx'
solData.to_excel(filename)

#data=pd.read_excel('heights.xlsx',skiprows=1)
##print (data.iloc[1,1])
#height=20000
##print (data)
#for i in range(len(h)):
#    if height>data.iloc[i,1]:
#        distance=(height-data.iloc[i,1])/(data.iloc[i+1,1]-data.iloc[i,1])
#        vgap=(data.iloc[i+1,2]-data.iloc[i,2])
#        v=vgap*distance+data.iloc[i,2]
#        break
#print (v)
        
