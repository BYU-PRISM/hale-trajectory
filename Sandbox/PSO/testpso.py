# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, tan, log10
from scipy.optimize import root
from dynamicsWrapper import uavDynamicsWrapper
from scipy.optimize import minimize
from pyswarm import pso

def banana(x):
    x1 = x[0]
    x2 = x[1]
    return [x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5]

def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [-3, -1]
ub = [2, 6]

xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)