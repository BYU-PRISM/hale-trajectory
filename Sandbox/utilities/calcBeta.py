# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
import yaml
import sys

# Choose folder
folder = 'hale_2017_11_06_13_57_36 - Beta Values Out'

# Get data
files = glob.glob('../../Data/'+folder+'/Intermediates/*.xlsx')
if(len(files)==0):
    print('No data files available for this folder.')
    sys.exit()
else:
    df = pd.read_excel(files[-1])

# Get Config File
config_file = glob.glob('../../Data/'+folder+'/*.yml')[-1]
with open(config_file, 'r') as ifile:
    config = yaml.load(ifile)
    
# Check if we're using wind
if(config['wind']['use_wind']==False):
    print('No Wind, beta = 0')
    sys.exit()
    
# Get position gradients
time_step = config['optimization']['time_step']['value']

dx_dt = np.gradient(df['x'],time_step)
dy_dt = np.gradient(df['y'],time_step)
dx = df['dx']
dy = df['dy']

# Wind vectors
w_n = config['wind']['w_n']
w_e = config['wind']['w_e']

# Heading
psi = np.mod(df['psi'],2*np.pi)

# Calculate sideslip
V_bar_g = np.array([dx,dy])
V_bar_w = np.array([[w_n],[w_e]])
V_bar_a = V_bar_g - V_bar_w
V_bar_a_n = V_bar_a[0,:]
V_bar_a_e = V_bar_a[1,:]
V_a_angle = np.arctan2(V_bar_a_e,V_bar_a_n)
V_a_angle[V_a_angle<0] = V_a_angle[V_a_angle<0]+2*np.pi
beta = V_a_angle - psi

beta2 = np.arccos((V_bar_a_e * np.sin(psi) + V_bar_a_n * np.cos(psi))/(np.sqrt(V_bar_a_n**2+V_bar_a_e**2)*np.sqrt(np.cos(psi)**2+np.sin(psi)**2)))

# Remove outliers
#m = 3
#beta = beta[abs(beta - np.mean(beta)) < m * np.std(beta)]
#beta = beta[abs(beta - np.mean(beta)) < m * np.std(beta)]

import matplotlib.pyplot as plt
plt.figure()
plt.plot(df.time,dx_dt)
plt.plot(df.time,dx,'--')
plt.title('dx')
plt.figure()
plt.plot(df.time,dy_dt)
plt.plot(df.time,dy,'--')
plt.title('dy')

plt.figure()
plt.plot(abs(np.degrees(beta)))
plt.plot(np.degrees(beta2),'--')
plt.title('Betas')

#plt.figure()
#plt.plot(np.degrees(beta))
#plt.title('Beta')
#plt.figure()
#plt.plot(np.degrees(beta2))
#plt.title('Beta2')
plt.figure()
plt.plot(df.time,np.degrees(V_a_angle),label='V_a')
plt.plot(df.time,np.degrees(psi),label='psi')
plt.legend()
plt.title('Va vs psi')
#
print('Max: ' + str(np.degrees(beta.max())))
print('Mean: ' + str(np.degrees(beta.mean())))
print('Median: ' + str(np.degrees(beta.median())))
print('Std: ' + str(np.degrees(beta.std())))