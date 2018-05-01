# -*- coding: utf-8 -*-
"""
Created on Tue May 02 14:56:00 2017

Updates:
    8-22-17 - removed P_int because it was being double-counted (and is the same as P_payload)

@author: UAV-PRISM
"""

from __future__ import division
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, tan, log10
from scipy.optimize import root
from dynamicsWrapper_wind import uavDynamicsWrapper_wind
from scipy.optimize import minimize
import sys


def power_objective(v,x0,h_0,w,smartsData,config_file):
    # Find Steady State at this v
    eq = root(uavDynamicsWrapper_wind,x0,method='lm',args=([],[],h_0,v,w,smartsData,config_file,2))
    if(eq.success==False):
        print('Could not find root in power objective')
        sys.exit()
    # Find power at this v
    P_N = uavDynamicsWrapper_wind(eq.x,[],[],h_0,v,w,smartsData,config_file,3)
    return [P_N]

# Constraints for power minimization
def power_constraints(v,x0,h_0,w,smartsData,config_file):
    # Find Steady State at this v
    eq = root(uavDynamicsWrapper_wind,x0,method='lm',args=([],[],h_0,v,w,smartsData,config_file,2))
    if(eq.success==False):
        print('Could not find root in power constraints')
        sys.exit()
    Tp = eq.x[0]
    phi = eq.x[2]
    # Find lift coefficient at this v
    cl = uavDynamicsWrapper_wind(eq.x,[],[],h_0,v,w,smartsData,config_file,4)
    constraint_array = [config_file['trajectory']['tp']['max'] - Tp,
                        Tp - config_file['trajectory']['tp']['min'],
                        config_file['trajectory']['phi']['max'] - phi,
                        phi - config_file['trajectory']['phi']['min'],
                        config_file['trajectory']['lift_coefficient']['max'] - cl]
    return constraint_array

def findSteadyState(x0,h_0,smartsData,config_file):
    '''
    For a given height, this returns the velocity, thrust, angle of attack,
    and bank angle for a minimum power level turn with the desired radius.
    '''
    config = config_file
    
    # Wind
    w = [config['wind']['w_n'],
         config['wind']['w_e'],
         config['wind']['w_d']]

    # Initial guess for velocity
    v_0 = config['trajectory']['v']['ss_initial_guess'] # 35
    
    cons = ({'type': 'ineq', 'fun':power_constraints,'args':(x0,h_0,w,smartsData,config_file)})
    
    sol = minimize(power_objective,
                   [v_0],
                   args=(x0,h_0,w,smartsData,config_file),
                   constraints=cons,
                   method='SLSQP',
                   options={'disp':True,'eps':1e-8,'ftol':1e-10})
#    sol = minimize(power_objective,
#                   [v_0],
#                   args=(x0,h_0,smartsData,config_file),
#                   method='SLSQP',
#                   options={'disp':True,'eps':1e-8,'ftol':1e-10})
    if(sol.success==False):
        print('Could not find minimum velocity')
        sys.exit()
    vmin = sol.x
    eq = root(uavDynamicsWrapper_wind,x0,method='lm',args=([],[],h_0,vmin,w,smartsData,config_file,2))
    Tpmin = eq.x[0]
    alphamin = eq.x[1]
    phimin = eq.x[2]
    clmin = uavDynamicsWrapper_wind(eq.x,[],[],h_0,vmin,w,smartsData,config_file,4)
    pmin = uavDynamicsWrapper_wind(eq.x,[],[],h_0,vmin,w,smartsData,config_file,3)
    
#    from findSteadyState_config import uavSSPower
#    from findSteadyState_config import uavSSdynamics
#    
#    Plist = []
#    vlist = []
#    TpList = []
#    alphaList = []
#    phiList = []
#    clList = []
##    for v_01 in np.arange(20,70,0.01):
##        eq = root(uavSSdynamics,x0,method='lm',args=(h_0,v_01,config_file))
##        P_N = uavSSPower(eq.x,h_0,v_01,config_file)
##        Plist.append(P_N)
##        vlist.append(v_01)
##        TpList.append(eq.x[0])
##        alphaList.append(eq.x[1])
##        phiList.append(eq.x[2])
##        clList.append((1.52-0.717)/(10-0)*(eq.x[1]*180/pi-0)+0.717)
#    
#    # Plots for comparison
#    Plist = []
#    vlist = []
#    TpList = []
#    alphaList = []
#    phiList = []
#    clList = []
#    
#    Plist2 = []
#    vlist2 = []
#    clList2 = []
#    TpList2 = []
#    alphaList2 = []
#    phiList2 = []
#    for v_sweep in np.arange(20,70,0.01):
#        # Old Version
#        eq1 = root(uavSSdynamics,x0,method='lm',args=(h_0,v_sweep,config_file))
#        if(eq1.success==False):
#            print('Could not find root in old velocity sweep')
#            sys.exit()
#        P_N = uavSSPower(eq1.x,h_0,v_sweep,config_file)
#        Plist.append(P_N)
#        vlist.append(v_sweep)
#        TpList.append(eq1.x[0])
#        alphaList.append(eq1.x[1])
#        phiList.append(eq1.x[2])
#        clList.append((1.52-0.717)/(10-0)*(eq1.x[1]*180/pi-0)+0.717)
#        
#        # New Version
#        eq2 = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,vmin,smartsData,config_file,2))
#        if(eq2.success==False):
#            print('Could not find root in new velocity sweep')
#            sys.exit()
##        P_N = uavDynamicsWrapper(eq.x,[],[],h_0,v_sweep,smartsData,config_file,3)
#        P_N = power_objective(v_sweep,x0,h_0,smartsData,config_file)
#        cl = uavDynamicsWrapper(eq2.x,[],[],h_0,v_sweep,smartsData,config_file,4)
#        Plist2.append(P_N)
#        vlist2.append(v_sweep)
#        clList2.append(cl)
#        TpList2.append(eq2.x[0])
#        alphaList2.append(eq2.x[1])
#        phiList2.append(eq2.x[2])
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    plt.plot(vlist2,Plist2)
#    plt.plot(vlist,Plist)
#    plt.title('v vs power new version')
#    plt.ylim([4000,16000])
#    fig = plt.figure()
#    plt.plot(vlist2,clList2)
#    plt.plot(vlist,clList)
#    plt.ylim([0,3.5])
#    plt.title('v vs cl new version')
#    fig = plt.figure()
#    plt.plot(vlist2,TpList2)
#    plt.plot(vlist,TpList)
#    plt.ylim([120,260])
#    plt.title('v vs Tp new version')
#    fig = plt.figure()
#    plt.plot(vlist2,alphaList2)
#    plt.plot(vlist,alphaList)
#    plt.ylim([-0.2,0.5])
#    plt.title('v vs alpha new version')
#    fig = plt.figure()
#    plt.plot(vlist2,phiList2)
#    plt.plot(vlist,phiList)
#    plt.ylim([0,0.18])
#    plt.title('v vs phi new version')
    
    return vmin,Tpmin,alphamin,phimin,clmin,pmin

if __name__ == "__main__":
    findSteadyState(x0,h_0,smartsData,config_file)