# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_excel('dataLong.xlsx')

def turningpoints(x):
  N=0
  i_list = []
  for i in range(1, len(x)-1):
     if ((x[i-1] < x[i] and x[i+1] < x[i]) 
         or (x[i-1] > x[i] and x[i+1] > x[i])):
       N += 1
       i_list.append(i)
  return i_list

def find_orbits(data):
    
    # Grab data
    x = data['x']
    y = data['y']
    z = data['h']
    t = data['time']
    
    # Find the points where the sin wave changes directions
    turn = turningpoints(x)
    
    # Plot wave with turning points
#    plt.figure()
#    plt.plot(x)
#    plt.scatter(turn,x[turn])
    
    diff = np.array(turn[1:]) - np.array(turn[:-1])
    
    np.extract(diff>10,turn[1:])
    
    turn_clip = np.r_[turn[0],np.extract(diff>10,turn[1:])]
    
#    plt.scatter(turn_clip,x[turn_clip])
    
#    plt.close('all')
#    plt.figure()
#    plt.plot(x,y)
#    plt.scatter(x[turn_clip],y[turn_clip])
    
    x_orbits = []
    y_orbits = []
    z_orbits = []
    t_orbits = []
    for i in range(1,38,2):
#        plt.figure()
#        plt.plot(x[turn_clip[i]:turn_clip[i+2]],y[turn_clip[i]:turn_clip[i+2]])
#        print(len(x[turn_clip[i]:turn_clip[i+2]]))
        x_orbits.append(x[turn_clip[i]:turn_clip[i+2]])
        y_orbits.append(y[turn_clip[i]:turn_clip[i+2]])
        z_orbits.append(z[turn_clip[i]:turn_clip[i+2]])
        t_orbits.append(t[turn_clip[i]])
        
    return x_orbits, y_orbits, z_orbits, t_orbits

def find_orbitsMV(data):
    
    # Grab data
    alpha = data['alpha']
    tp = data['tp']
    phi = data['phi']
    p_bat = data['p_bat']
    t = data['time']
    az = data['azimuth']
    zen = data['zenith']
    x = data['x']
    
    # Find the points where the sin wave changes directions
    turn = turningpoints(x)
    
    # Plot wave with turning points
#    plt.figure()
#    plt.plot(x)
#    plt.scatter(turn,x[turn])
    
    diff = np.array(turn[1:]) - np.array(turn[:-1])
    
    np.extract(diff>10,turn[1:])
    
    turn_clip = np.r_[turn[0],np.extract(diff>10,turn[1:])]
    
#    plt.scatter(turn_clip,x[turn_clip])
    
#    plt.close('all')
#    plt.figure()
#    plt.plot(x,y)
#    plt.scatter(x[turn_clip],y[turn_clip])
    
    alpha_orbits = []
    tp_orbits = []
    phi_orbits = []
    p_bat_orbits = []
    t_orbits = []
    az_orbits = []
    zen_orbits = []
    for i in range(1,len(turn_clip)-1,2):
#        plt.figure()
#        plt.plot(x[turn_clip[i]:turn_clip[i+2]],y[turn_clip[i]:turn_clip[i+2]])
#        print(len(x[turn_clip[i]:turn_clip[i+2]]))
        alpha_orbits.append(alpha[turn_clip[i]:turn_clip[i+2]])
        tp_orbits.append(tp[turn_clip[i]:turn_clip[i+2]])
        phi_orbits.append(phi[turn_clip[i]:turn_clip[i+2]])
        p_bat_orbits.append(p_bat[turn_clip[i]:turn_clip[i+2]])
        t_orbits.append(t[turn_clip[i]])
        az_orbits.append(az[turn_clip[i]])
        zen_orbits.append(zen[turn_clip[i]])
        
    return alpha_orbits,tp_orbits,phi_orbits,p_bat_orbits,t_orbits,az_orbits,zen_orbits

if __name__ == '__main__':
    data = pd.read_excel('dataNew.xlsx')
    find_orbitsMV(data)