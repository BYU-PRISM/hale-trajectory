# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, sqrt, cos, sin, exp, tan, log10, arcsin
import pandas as pd

v_g_0 = 35
gamma_0 = 0
chi_0 = 0
h_0 = 18288
x0 = 0
y0 = 0
E_Batt_0 = 50

SV0 = [v_g_0,gamma_0,chi_0,h_0,x0,y0,E_Batt_0]

Tpmin = 100
alphamin = 0.01
phimin = 0

MV0 = [Tpmin,alphamin,phimin]

# Load SVs
v_g,gamma,chi,h,x,y,E_Batt = SV0
# Load MVs
Tp, alpha, phi = MV0

t = 0

g = 9.80665 # Gravity (m/s**2)
R_air = 287.041 # Gas Constant for air (m**2/(s**2 K))
rho_11 = 0.364 # Standard air density at 11 km (kg/m**3)
T_11 = 216.66 # Standard air temp at 11 km (K)
mu = 1.4397e-5 # Air viscosity from 11 km - 25 km (Standard Atmosphere)

# Masses and Densities
m = config['aircraft']['mass_total']['value'] # 425.0 # Total Mass (kg) (FB)
#    E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
#    m_battery = config['aircraft']['battery']['mass']['value'] # 212.0 # (kg) (FB)

# Wind
w_n = -1
w_e = 0
w_d = 0
v_w = sqrt(w_n**2 + w_e**2 + w_d**2) # Magnitidue of wind speed
v_a = sqrt(v_g**2-2*v_g*(w_n*cos(chi)*cos(gamma)+w_e*sin(chi)*cos(gamma)-w_d*sin(gamma))+v_w**2) # Positive root    
gamma_a = arcsin((v_g*sin(gamma) + w_d)/v_a) #WIND
psi = chi - arcsin((-w_n*sin(chi) + w_e*cos(chi))/(v_a*cos(gamma_a))) #WIND    

# Drag Model (Translated from Judd's Matlab code)
#    Lambda = 20*pi/180.0 # Sweep angle (Judd)
AR = config['aircraft']['aspect_ratio'] # 30.0 # Aspect ratio (FB)
#    thickness = config['aircraft']['airfoil_thickness_ratio'] # 0.11 # Thickness ratio (Judd)
S = config['aircraft']['wing_top_surface_area']['value'] # 60.0 # Wing and solar panel area (m**2) (FB)
b = sqrt(S*AR) # Wingspan (m)
chord = b/AR # Chord (m)
#    roughness = config['aircraft']['roughness_factor'] # 1.1 # Roughness factor (Judd)
es = config['aircraft']['inviscid_span_efficiency'] # 0.98 # Inviscid span efficiency (Judd)
#    xcrit_top = config['aircraft']['xcrit_top'] # 0.7412 # Top surface transition point (Judd)
#    xcrit_bottom = config['aircraft']['xcrit_bottom'] # 1.0 # Bottom surface transition point (Judd)
#    S_wet = (1+0.2*thickness)*S # Wetted surface area for one side (m**2)
#    Ck = config['aircraft']['ck'] # 1.1 # empirical constant used in finding form factor
#    k = 1 + 2*Ck*thickness*(cos(Lambda))**2 + Ck**2*(cos(Lambda)**2*thickness**2*(1+5*(cos(Lambda)**2)))/2.0 # Form Factor

# Propeller Efficiency
R_prop = config['aircraft']['propeller_radius']['value'] # 2.0 # Propeller Radius (m) - Kevin

# Power
e_motor = config['aircraft']['motor_efficiency'] # 0.95 # Efficiency of motor
P_payload = config['aircraft']['power_for_payload']['value'] # 250.0 # Power for payload (W)
#    E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
panel_efficiency = config['solar']['panel_efficiency'] # 0.25 # (FB)

# Manipulated variables
Tp = Tp_0 #3.171 # 48.4 # Thrust (N) (function input)
alpha = alpha_0 # Angle of Attack (rad) (function input)
#phi = phi_0 #0.038 #2.059E-03 # 0.0001 # Bank Angle (rad) (function input)

#### Atmospheric Effects
rho = rho_11 * exp(-(g/(R_air*T_11))*(h-11000)) # Air density (kg/m**3)

# Pitch from AoA
theta = gamma + alpha

# Alpha in degress for fits
alpha_deg = np.degrees(alpha)

# CL from AoA, Lift slope line from xflr5
#    cl = (1.59-0.656)/(10-0)*(alpha*180/pi-0)+0.656
cl = 0.0925*alpha_deg + 0.6613

#### Drag Model
### Top Surface
## Top Reynolds Numbers
# Flat plate Reynolds number
Re = rho*v_a*chord/mu # Reynolds number

# New viscous/parasitic drag from xflr5
C_D_p = 6.2740486643e-07*alpha_deg**5 - 1.4412023268e-05*alpha_deg**4 + 1.1160259529e-04*alpha_deg**3 - 2.2563681473E-04*alpha_deg**2 - 8.6114793593E-05*alpha_deg + 1.0569281079E-02

# Oswald efficiency factor
k_e = 0.4*C_D_p
e_o = 1/((pi*AR*k_e) + (1/es))
# Drag coefficient
C_D = C_D_p + cl**2/(pi*AR*e_o)

#### Flight Dynamics
q = 1/2.0*rho*v_a**2 # Dynamic pressure (Pa)

L = q*cl*S # Lift (N) (simplified definition using q)
D = C_D*q*S # Corrected Drag (N)

nh = L*sin(phi)/(m*g) # Horizontal load factor
nv = L*cos(phi)/(m*g) # Vertical load factor

### Propeller Max Theoretical Efficiency
Adisk = pi * R_prop**2 # Area of disk
e_prop = 2.0 / (1.0 + ( D / (Adisk * v_a**2.0 * rho/2.0) + 1.0 )**0.5)
nu_prop = e_prop * e_motor

#### Power
P_N = P_payload + v_a*Tp/nu_prop # Power Needed by Aircraft


# Flight Dynamics - with WIND
dv_g_dt = ((Tp-D)/(m*g)-sin(gamma))*g
dgamma_dt = g/v_g*(nv-cos(gamma))
dchi_dt = g/v_g*(nh/cos(gamma))*cos(chi-psi)
dh_dt = v_g*sin(gamma)
dx_dt = v_g*cos(chi)*cos(gamma)
dy_dt = v_g*sin(chi)*cos(gamma)
dist = (sqrt(x**2+y**2))
radius = v_g**2/(g*tan(phi))# Flight path radius