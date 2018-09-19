# -*- coding: utf-8 -*-
"""
Created on Tue May 02 14:56:00 2017

Updates:
    8-22-17 - removed P_int because it was being double-counted (and is the same as P_payload)

@author: UAV-PRISM
"""

from __future__ import division
import numpy as np

from dynamics import uavDynamics
from scipy.optimize import minimize

import sys


def ss_objective(xv,h_0,smartsData,config_file):
    x0 = xv[:3]
    v = xv[3]
    P_N = uavDynamics(x0,[],[],h_0,v,smartsData,config_file,3)
    return P_N

def ss_constraints(xv,h_0,smartsData,config_file):
    x0 = xv[:3]
    v = xv[3]
    d = uavDynamics(x0,[],[],h_0,v,smartsData,config_file,2)
    return d

def findSteadyState(x0,h_0,smartsData,config_file):
    '''
    For a given height, this returns the velocity, thrust, angle of attack,
    and bank angle for a minimum power level turn with the desired radius.
    '''
    config = config_file
#
#    # Initial guess for velocity
    v_0 = config['trajectory']['v']['ss_initial_guess'] # 35

    cons = ({'type': 'eq', 'fun':ss_constraints,'args':(h_0,smartsData,config_file)})
    
    bounds = [(config_file['trajectory']['tp']['min'],config_file['trajectory']['tp']['max']),
               (config_file['trajectory']['phi']['min'],config_file['trajectory']['phi']['max']),
               (config_file['trajectory']['alpha']['min'],config_file['trajectory']['alpha']['max']),
               (0,100)]
    
    # Find dynamic equlibrium
    x0v0 = np.r_[x0,v_0]
    sol = minimize(ss_objective,
                   [x0v0],
                   args=(h_0,smartsData,config_file),
                   bounds = bounds,
                   constraints=cons,
                   method='SLSQP',
                   options={'disp':True,'eps':1e-8,'ftol':1e-8})

    if(sol.success==False):
        print('Could not find minimum velocity')
        sys.exit()
    Tpmin = sol.x[0]
    alphamin = sol.x[1]
    phimin = sol.x[2]
    vmin = sol.x[3]
    clmin = uavDynamics(sol.x[:3],[],[],h_0,vmin,smartsData,config_file,4)
    pmin = sol.fun
    
    return vmin,Tpmin,alphamin,phimin,clmin,pmin