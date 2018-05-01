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
from dynamicsWrapperSM import uavDynamicsWrapper
from scipy.optimize import minimize
from pyswarm import pso
import sys


def power_objective(v,x0,h_0,smartsData,config_file):
    # Find Steady State at this v
    x0 = np.asarray(x0) * (v/32.0) # Adjust initial guesses
    methods = ['hybr','lm','broyden1','broyden2','anderson','linearmixing','diagbroyden','excitingmixing','krylov','df-sane']
    for method in methods:
        eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v,smartsData,config_file,2),tol=1e-4)
        if(eq.success==True):
#            print(method)
            break
#    eq = root(uavDynamicsWrapper,x0,args=([],[],h_0,v,smartsData,config_file,2))
    if(eq.success==False):
        print('Could not find root in power objective')
        sys.exit()
#        print('Reinitializing')
#        lb = np.asarray(x0) * 0.5
#        ub = np.asarray(x0) * 2
#        x0, fopt = pso(pso_root, lb=lb, ub=ub, args=(v_0,h_0,smartsData,config_file),maxiter=1000,swarmsize=300,minstep=1e-10)
#        eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v,smartsData,config_file,2))
#        if(eq.success==False):
#            print('Still could not find root in power objective')
#            sys.exit()
    # Find power at this v
    P_N = uavDynamicsWrapper(eq.x,[],[],h_0,v,smartsData,config_file,3)
    return [P_N]

def pso_root(x0,v,h_0,smartsData,config_file):
    d = uavDynamicsWrapper(x0,[],[],h_0,v,smartsData,config_file,2)
    return np.sum(np.abs(d))

def pso_root_v(x0,h_0,smartsData,config_file):
    v = x0[-1]
    x0 = x0[:-1]
    d = uavDynamicsWrapper(x0,[],[],h_0,v,smartsData,config_file,2)
    P_N = uavDynamicsWrapper(x0,[],[],h_0,v,smartsData,config_file,3)
    return np.sum(np.abs(d)) + P_N

# Constraints for power minimization
def power_constraints(v,x0,h_0,smartsData,config_file):
    # Find Steady State at this v
    x0 = np.asarray(x0) * (v/32.0) # Adjust initial guesses
#    eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v,smartsData,config_file,2))
    methods = ['hybr','lm','broyden1','broyden2','anderson','linearmixing','diagbroyden','excitingmixing','krylov','df-sane']
    for method in methods:
        eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v,smartsData,config_file,2),tol=1e-4)
        if(eq.success==True):
#            print(method)
            break
#    eq = root(uavDynamicsWrapper,x0,args=([],[],h_0,v,smartsData,config_file,2))
    if(eq.success==False):
        print('Could not find root in power constraints')
        sys.exit()
#        print('Reinitializing')
#        lb = np.asarray(x0) * 0.5
#        ub = np.asarray(x0) * 2
#        x0, fopt = pso(pso_root, lb=lb, ub=ub, args=(v_0,h_0,smartsData,config_file),maxiter=1000,swarmsize=300,minstep=1e-10)
#        eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v,smartsData,config_file,2))
#        if(eq.success==False):
#            print('Still could not find root in power constraints')
#            sys.exit()
    Tp = eq.x[0]
    alpha = eq.x[1]
    phi = eq.x[2]
    # Find lift coefficient at this v
    constraint_array = [config_file['trajectory']['tp']['max'] - Tp,
                        Tp - config_file['trajectory']['tp']['min'],
                        config_file['trajectory']['phi']['max'] - phi,
                        phi - config_file['trajectory']['phi']['min'],
                        config_file['trajectory']['alpha']['max'] - alpha,
                        alpha - config_file['trajectory']['alpha']['min']]
    return constraint_array

def findSteadyState(x0,h_0,v_0,smartsData,config_file):
    '''
    For a given height, this returns the velocity, thrust, angle of attack,
    and bank angle for a minimum power level turn with the desired radius.
    '''
    config = config_file

    # Initial guess for velocity
#    v_0 = config['trajectory']['v']['ss_initial_guess'] * h_0/18288 # 35
    
    # New guess values from PSO
    lb = [config_file['trajectory']['tp']['min'],
          config_file['trajectory']['alpha']['min'],
          config_file['trajectory']['phi']['min']]
#    lb = np.asarray(x0) * 0.5
    
    ub = [config_file['trajectory']['tp']['max']*1.5,
          config_file['trajectory']['alpha']['max'],
          config_file['trajectory']['phi']['max']]
#    ub = np.asarray(x0) * 5
    
    x0, fopt = pso(pso_root, lb=lb, ub=ub, args=(v_0,h_0,smartsData,config_file),maxiter=1000,swarmsize=100)
    print('New Guess Values: ' +str(x0))
    
    cons = ({'type': 'ineq', 'fun':power_constraints,'args':(x0,h_0,smartsData,config_file)})
    
    sol = minimize(power_objective,
                   [v_0],
                   args=(x0,h_0,smartsData,config_file),
                   constraints=cons,
                   method='SLSQP',
                   options={'disp':True,'eps':1e-8,'ftol':1e-8})

    if(sol.success==False):
        print('Could not find minimum velocity. Retrying')
        x0, fopt = pso(pso_root, lb=lb, ub=ub, args=(v_0,h_0,smartsData,config_file),maxiter=1000,swarmsize=1000)
        print('New Guess Values: ' +str(x0))
        cons = ({'type': 'ineq', 'fun':power_constraints,'args':(x0,h_0,smartsData,config_file)})
    
        sol = minimize(power_objective,
                       [v_0],
                       args=(x0,h_0,smartsData,config_file),
                       constraints=cons,
                       method='SLSQP',
                       options={'disp':True,'ftol':1e-4})
        if(sol.success==False):
            print('Could not find minimum velocity. Retrying')
            x0, fopt = pso(pso_root, lb=lb, ub=ub, args=(v_0,h_0,smartsData,config_file),maxiter=1000,swarmsize=2500)
            print('New Guess Values: ' +str(x0))
            cons = ({'type': 'ineq', 'fun':power_constraints,'args':(x0,h_0,smartsData,config_file)})
        
            sol = minimize(power_objective,
                           [v_0],
                           args=(x0,h_0,smartsData,config_file),
                           constraints=cons,
                           method='SLSQP',
                           options={'disp':True,'ftol':1e-2})
            if(sol.success==False):
                print('Could Not find minimum velocity.  Failed.')
                sys.exit()
    vmin = sol.x
    eq = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,vmin,smartsData,config_file,2))
    Tpmin = eq.x[0]
    alphamin = eq.x[1]
    phimin = eq.x[2]
    clmin = uavDynamicsWrapper(eq.x,[],[],h_0,vmin,smartsData,config_file,4)
    pmin = uavDynamicsWrapper(eq.x,[],[],h_0,vmin,smartsData,config_file,3)
    
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
##        # Old Version
##        eq1 = root(uavSSdynamics,x0,method='lm',args=(h_0,v_sweep,config_file))
##        if(eq1.success==False):
##            print('Could not find root in old velocity sweep')
##            sys.exit()
##        P_N = uavSSPower(eq1.x,h_0,v_sweep,config_file)
##        Plist.append(P_N)
##        vlist.append(v_sweep)
##        TpList.append(eq1.x[0])
##        alphaList.append(eq1.x[1])
##        phiList.append(eq1.x[2])
##        clList.append((1.52-0.717)/(10-0)*(eq1.x[1]*180/pi-0)+0.717)
##        
#        # New Version
#        eq2 = root(uavDynamicsWrapper,x0,method='lm',args=([],[],h_0,v_sweep,smartsData,config_file,2))
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
#    plt.scatter(vmin,pmin)
##    plt.plot(vlist,Plist)
#    plt.title('v vs power new version')
#    plt.ylim([4000,16000])
#    fig = plt.figure()
#    plt.plot(vlist2,clList2)
##    plt.plot(vlist,clList)
#    plt.ylim([0,3.5])
#    plt.title('v vs cl new version')
#    fig = plt.figure()
#    plt.plot(vlist2,TpList2)
##    plt.plot(vlist,TpList)
#    plt.ylim([90,260])
#    plt.title('v vs Tp new version')
#    fig = plt.figure()
#    plt.plot(vlist2,alphaList2)
##    plt.plot(vlist,alphaList)
#    plt.ylim([-0.2,0.5])
#    plt.title('v vs alpha new version')
#    fig = plt.figure()
#    plt.plot(vlist2,phiList2)
##    plt.plot(vlist,phiList)
#    plt.ylim([0,0.18])
#    plt.title('v vs phi new version')
    
    return vmin,Tpmin,alphamin,phimin,clmin,pmin

if __name__ == "__main__":
    vmin,Tpmin,alphamin,phimin,clmin,pmin = findSteadyState(x0,h_0,smartsData,config_file)