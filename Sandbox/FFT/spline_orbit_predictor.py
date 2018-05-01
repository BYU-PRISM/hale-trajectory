# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
from findOrbits import find_orbitsMV

# Load trajectory data
data = pd.read_excel('dataNew.xlsx')

#data = data.iloc[0:300]

# Split into orbits
alpha_orbits,tp_orbits,phi_orbits,p_bat_orbits,t_orbits,az_orbits,zen_orbits = find_orbitsMV(data)

# Take spline of each orbit individually - alpha
alpha_knots = []
alpha_coeffs = []
for alpha in alpha_orbits:
    s = splrep(range(len(alpha)),alpha)
    alpha_knots.append(s[0])
    alpha_coeffs.append(s[1])