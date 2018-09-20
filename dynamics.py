from __future__ import division
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, tan
from solar_functions import solarFlux

def uavDynamics(a1,a2,a3,h_0,v_0,config,mode):
    '''
    This function makes it possible to use the same model for root finding,
    integration, power calculations and post processing
    
    Integration
    sol,output = odeint(uavDynamicsWrapper, SV0, t, args=(MV,h_0,[],smartsData,config,1))
    
    Root Finding
    eq = root(uavDynamicsWrapper,MV0,method='lm',args=([],[],h_0,v_0,smartsData,config,2))
    
    Power Required
    p_n = uavDynamicsWrapper(x_eq,[],[],h_0,v_0,smartsData,config,3)
    
    Lift Coefficient
    cl = uavDynamicsWrapper(x_eq,[],[],h_0,v_0,smartsData,config,4)
    '''
    
    config = config
    
    ## Process Inputs
    if(mode==1):
        # Integration
        # Here we need to pass in the initial state variable conditions and
        # the time to odeint.  We also pass in a fixed set of MVs.
        SV = a1
        t = a2
        MV = a3
        # Load SVs
        v,gamma,psi,h,x,y,E_Batt = SV
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        
    if(mode==2 or mode==3 or mode==4):
        # Root Finding, Power
        # In the root finding case, MV is the initial guesses.  In the Power
        # or cl case, it is a fixed set of MVs
        MV = a1
        t = 0 # time is arbitrary in this case
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        # Load SVs
        v = v_0
        h = h_0
        gamma = config.aircraft.gamma.level
        psi = config.aircraft.psi.initial_value
        initial_SOC = config.aircraft.battery_initial_SOC.value
        E_d = config.aircraft.battery_energy_density.value # Battery energy density (W*hr/kg) (FB)
        m_battery = config.aircraft.mass_battery.value # Battery mass
        E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
        E_Batt = E_batmax*initial_SOC # Initial Battery Charge
        x = config.x.initial_value
        y = config.y.initial_value
        
    if(mode==5):
        # Output all other variables
        SV = a1
#        t = a2
        MV = a3
        # Load SVs
        v,gamma,psi,h,x,y,E_Batt,t = SV
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        pass
        
    ## Run Model
    m = model(t,v,gamma,psi,h,x,y,E_Batt,Tp_0,alpha_0,phi_0,config,mode)
        
    ## Process Outputs
    if(mode==1):
        # Integration
        output = [
         m['dv_dt'],
         m['dgamma_dt'],
         m['dpsi_dt'],
         m['dh_dt'],
         m['dx_dt'],
         m['dy_dt'],
         m['dE_Batt_dt']
         ]
        
    if(mode==2):
        # Root Finding
        output = [
         m['dv_dt'],
         m['dgamma_dt'],
         m['radius_const']
         ]
        output = np.ravel(output)
    
    if(mode==3):
        # Power
        output = m['P_N']
        
    if(mode==4):
        # cl
        output = m['cl']
        
    if(mode==5):
        # Output all other variables
        output = m
        
    return output

def model(t,v,gamma,psi,h,x,y,E_batt,Tp_0,alpha_0,phi_0,config,mode):
    '''
    Inputs
    
    '''
    
    config = config
    
    # Time
    # Cycle back to beginning if past 24 hours
    if(t>3600*24):
        t = t-3600*24
    
    # Atmospheric Effects
    g = 9.80665 # Gravity (m/s**2)
    R_air = 287.041 # Gas Constant for air (m**2/(s**2 K))
    rho_11 = 0.364 # Standard air density at 11 km (kg/m**3)
    T_11 = 216.66 # Standard air temp at 11 km (K)
    
    # Masses and Densities
    m = config.aircraft.mass_total.value # 425.0 # Total Mass (kg) (FB)
    E_d = config.aircraft.battery_energy_density.value # 350.0 # Battery energy density (W*hr/kg) (FB)
    m_battery = config.aircraft.mass_battery.value # 212.0 # (kg) (FB)
    
    S = config.aircraft.wing_top_surface_area.value # 60.0 # Wing and solar panel area (m**2) (FB)
    chord = config.aircraft.chord.value # Chord (m)
    
    
    # Propeller Efficiency
    R_prop = config.aircraft.propeller_radius.value # 2.0 # Propeller Radius (m) - Kevin

    # Power
    e_motor = config.aircraft.motor_efficiency.value # 0.95 # Efficiency of motor
    P_payload = config.aircraft.power_for_payload.value # 250.0 # Power for payload (W)
    E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
    
    # Manipulated variables
    Tp = Tp_0 #3.171 # 48.4 # Thrust (N) (function input)
    alpha = alpha_0 # Angle of Attack (rad) (function input)
    phi = phi_0 #0.038 #2.059E-03 # 0.0001 # Bank Angle (rad) (function input)
    
    #### Atmospheric Effects
    rho = rho_11 * exp(-(g/(R_air*T_11))*(h-11000)) # Air density (kg/m**3)
    T_air = -56.46 + 273.15 # Air temperature (K)
    mu = 1.458e-6*sqrt(T_air)/(1+110.4/T_air) # Dynamic viscosity of air (kg/(m s))
    
    # Pitch from AoA
    theta = gamma + alpha
    
    # Alpha in degress for fitted functions
    alpha_deg = np.degrees(alpha)
    
    # Flat plate Reynolds number
    Re = rho*v*chord/mu # Reynolds number
    
    # CL from emperical fit
    cl = config.aircraft.CL(alpha_deg,Re)
    
    # CD from emperical fit
    C_D = config.aircraft.CD(alpha_deg,Re)
    
    #### Flight Dynamics
    q = 1/2.0*rho*v**2 # Dynamic pressure (Pa)
    
    L = q*cl*S # Lift (N) (simplified definition using q)
    D = C_D*q*S # Corrected Drag (N)
    
    nh = L*sin(phi)/(m*g) # Horizontal load factor
    nv = L*cos(phi)/(m*g) # Vertical load factor
    
    ### Propeller Max Theoretical Efficiency
    Adisk = pi * R_prop**2 # Area of disk
    e_prop = 2.0 / (1.0 + ( Tp / (Adisk * v**2.0 * rho/2.0) + 1.0 )**0.5)
    nu_prop = e_prop * e_motor
    
    #### Power
    P_N = P_payload + v*Tp/nu_prop # Power Needed by Aircraft
    
    if(mode==1 or mode==5):
        solar_data = solarFlux(config.solar.smartsData, t/3600.0, phi,theta, psi)
        G_sol = solar_data[0][0]
        zenith = solar_data[0][1]
        azimuth = solar_data[0][2]
        h_flux = solar_data[0][3]
        mu_solar = solar_data[0][4]
        sn1 = solar_data[0][5]
        sn2 = solar_data[0][6]
        sn3 = solar_data[0][7]
        flux = solar_data[0][8]
    else:
        # These aren't required to find steady state flight or power needs
        G_sol = 0
        zenith = 0
        azimuth = 0
        h_flux = 0
        mu_solar = 0
        sn1 = 0
        sn2 = 0
        sn3 = 0
        flux = 0
        
    # Solar Efficiency
    panel_efficiency = config.aircraft.panel_efficiency(G_sol)
    P_solar = G_sol*S*panel_efficiency
    
    # Flight Dynamics
    dv_dt = ((Tp-D)/(m*g)-sin(gamma))*g
    dgamma_dt = g/v*(nv-cos(gamma))
    dpsi_dt = g/v*(nh/cos(gamma))
    dh_dt = v*sin(gamma)
    dx_dt = v*cos(psi)*cos(gamma)
    dy_dt = v*sin(psi)*cos(gamma)
    dist = (sqrt(x**2+y**2))
    radius = v**2/(g*tan(phi))# Flight path radius
    
    radius_max = config.x.max # 3000 (m)
    
    # Power
    P_bat = P_solar - P_N # Power used to charge or discharge battery (W)
    dE_Batt_dt = P_bat*1e-6 # (Convert to MJ)
    h_0 = config.h.initial_value
    TE = E_batt + m*g*(h-h_0)*1e-6
    
    # Clip d_E_batt at maximum battery
    bat_switch = 1/(1+exp(-10*(E_batmax - E_batt)))
    if(dE_Batt_dt>0):
        dE_Batt_dt = dE_Batt_dt * bat_switch
    
    # Collect model outputs into dictionary
    model_output = {"dv_dt":dv_dt,
                    "dgamma_dt":dgamma_dt,
                    "dpsi_dt":dpsi_dt,
                     "dh_dt":dh_dt,
                     "dx_dt":dx_dt,
                     "dy_dt":dy_dt,
                     "dE_Batt_dt":dE_Batt_dt,
                     "radius_const":radius-radius_max,
                     "P_N":P_N,
                     'cl':cl,
                     'mu_solar':mu_solar,
                     'flux':flux,
                     'sn1':sn1,
                     'sn2':sn2,
                     'sn3':sn3,
                     'azimuth':azimuth,
                     'zenith':zenith,
                     'sun_h':h_flux,
                     'g_sol':G_sol,
                     'panel_efficiency':panel_efficiency,
                     'p_solar':P_solar,
                     'p_n':P_N,
                     'p_bat':P_bat,
                     'd':D,
                     'cd':C_D,
                     'cl':cl,
                     'rho':rho,
                     'm':m,
                     'nh':nh,
                     'nv':nv,
                     'nu_prop':nu_prop,
                     'dist':dist,
                     'theta':theta,
                     'te':TE,
                     're':Re,
                     'mu':mu}
    
    return model_output