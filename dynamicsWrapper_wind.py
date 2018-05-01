from __future__ import division
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, tan, log10, arcsin
from CombinedSolar import solarFlux
import pandas as pd

def uavDynamicsWrapper_wind(a1,a2,a3,h_0,v_0,w,smartsData,config_file,mode):
    '''
    This function makes it possible to use the same model for root finding,
    integration, power calculations and post processing
    
    Integration
    sol,output = odeint(uavDynamicsWrapper, SV0, t, args=(MV,h_0,[],w,smartsData,config_file,1))
    
    Root Finding
    eq = root(uavDynamicsWrapper,MV0,method='lm',args=([],[],h_0,v_0,w,smartsData,config_file,2))
    
    Power Required
    p_n = uavDynamicsWrapper(x_eq,[],[],h_0,v_0,w,smartsData,config_file,3)
    
    Lift Coefficient
    cl = uavDynamicsWrapper(x_eq,[],[],h_0,v_0,w,smartsData,config_file,4)
    '''
    
    config = config_file
    
    ## Process Inputs
    if(mode==1):
        # Integration
        # Here we need to pass in the initial state variable conditions and
        # the time to odeint.  We also pass in a fixed set of MVs.
        SV = a1
        t = a2
        MV = a3
        # Load SVs
        v_a,gamma,chi,h,x,y,E_Batt = SV
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        
        
    if(mode==2 or mode==3 or mode==4):
        # Root Finding, Power, cl
        # In the root finding case, MV is the initial guesses.  In the Power
        # or cl case, it is a fixed set of MVs
        # Set wind = 0
        w =[0,0,0]
        MV = a1
        t = 0 # time is arbitrary in this case
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        # Load SVs
        v_a = v_0
        h = h_0
        gamma = config['trajectory']['gamma']['initial_value']
        chi = 0
        initial_SOC = config['aircraft']['battery']['initial_state_of_charge']
        E_d = config['aircraft']['battery']['energy_density']['value'] # Battery energy density (W*hr/kg) (FB)
        m_battery = config['aircraft']['battery']['mass']['value'] # Battery mass
        E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
        E_Batt = E_batmax*initial_SOC # Initial Battery Charge
        x = config['trajectory']['x']['initial_value']
        y = config['trajectory']['y']['initial_value']
        
    if(mode==5):
        # Output all other variables
        SV = a1
#        t = a2
        MV = a3
        # Load SVs
        v_a,gamma,chi,h,x,y,E_Batt,t = SV
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        
    if(mode==6):
        # Root Finding, Power, cl
        # In the root finding case, MV is the initial guesses.  In the Power
        # or cl case, it is a fixed set of MVs
        # Set wind = 0
        w =[0,0,0]
        MV = a1
        t = 0 # time is arbitrary in this case
        # Load MVs
        Tp_0, alpha_0, phi_0 = MV
        # Load SVs
        v_a = v_0
        h = h_0
        gamma = config['trajectory']['gamma']['initial_value']
        chi = 0
        initial_SOC = config['aircraft']['battery']['initial_state_of_charge']
        E_d = config['aircraft']['battery']['energy_density']['value'] # Battery energy density (W*hr/kg) (FB)
        m_battery = config['aircraft']['battery']['mass']['value'] # Battery mass
        E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
        E_Batt = E_batmax*initial_SOC # Initial Battery Charge
        x = config['trajectory']['x']['initial_value']
        y = config['trajectory']['y']['initial_value']
        
    ## Run Model
    m = model(t,v_a,gamma,chi,h,x,y,E_Batt,Tp_0,alpha_0,phi_0,w,smartsData,config_file,mode)
        
    ## Process Outputs
    if(mode==1):
        # Integration
        output = [
         m['dv_a_dt'],
         m['dgamma_dt'],
         m['dchi_dt'],
         m['dh_dt'],
         m['dx_dt'],
         m['dy_dt'],
         m['dE_Batt_dt']
         ]
        
    if(mode==2):
        # Root Finding
        output = [
         m['dv_g_dt'],
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
#        output = [
#         m['mu'],
#         m['flux'],
#         m['sn1'],
#         m['sn2'],
#         m['sn3'],
#         m['azimuth'],
#         m['zenith'],
#         m['sun_h'],
#         m['g_sol'],
#         m['panel_efficiency'],
#         m['p_solar'],
#         m['p_n'],
#         m['p_bat'],
#         m['d'],
#         m['cd'],
#         m['cl'],
#         m['rho'],
#         m['m'],
#         m['nh'],
#         m['nv'],
#         m['nu_prop'],
#         m['dist'],
#         m['theta'],
#         m['te']
#        ]
        output = m
        
#        if(mode==6):
#        # Root Finding Straight Path
#        output = [
#         m['dv_a_dt'],
#         m['dgamma_dt']
#         ]
#        output = np.ravel(output)
        
    return output

# Note
# Added w, wind_param, phi

def model(t,v_a,gamma,chi,h,x,y,E_batt,Tp_0,alpha_0,phi_0,w,smartsData,config_file,mode):
    '''
    Inputs
    
    '''
    
#    v_a = 44.616226529999999

    
    config = config_file
    
    # Time
    # Cycle back to beginning if past 24 hours
    if(t>3600*24):
        t = t-3600*24
    
    # Atmospheric Effects
    g = 9.80665 # Gravity (m/s**2)
    R_air = 287.041 # Gas Constant for air (m**2/(s**2 K))
    rho_11 = 0.364 # Standard air density at 11 km (kg/m**3)
    T_11 = 216.66 # Standard air temp at 11 km (K)
    mu = 1.4397e-5 # Air viscosity from 11 km - 25 km (Standard Atmosphere)
    
    # Masses and Densities
    m = config['aircraft']['mass_total']['value'] # 425.0 # Total Mass (kg) (FB)
    E_d = config['aircraft']['battery']['energy_density']['value'] # 350.0 # Battery energy density (W*hr/kg) (FB)
    m_battery = config['aircraft']['battery']['mass']['value'] # 212.0 # (kg) (FB)
    
    # Surface fit for drag vs alpha and Re
    if(config['aircraft']['name'] == 'Aquila'):
        dsa = 1.2379225459790737E+02
        dsb = -1.2385770684910669E+04
        dsc = -5.7633649503331696E-01
        dsd = -6.3984649510237453E-05
        dsf = 1.0013226926073848E+00
        dsg = 4.0573453980222610E+01
        dsh = -2.0332829937750183E+06
    elif(config['aircraft']['name'] == 'Helios'):
        dsa = 5.8328354781017406E+03
        dsb = -1.6309313517819710E+01
    elif(config['aircraft']['name'] == 'Aquila E216'):
        dsa = 9.6878064972771682E-02
        dsb = -1.1914394969415213E-02
        dsc = 1.4658946775121501E-07
        dsd = -7.4933620263012425E-09
        dsf = -1.6444247419782368E-01
        dsg = 3.9791780146017000E-05
        dsh = -4.1825694373660372E-06
    
    # Wind
    w_n = w[0] # Extract wind components
    w_e = w[1]
    w_d = w[2]
    v_w = sqrt(w_n**2 + w_e**2 + w_d**2) # Magnitidue of wind speed
    v_g = (w_n*cos(chi)+w_e*sin(chi))+np.sqrt((w_n*cos(chi)+w_e*sin(chi))**2-v_w**2+v_a**2)
    v_g_actual = v_g
#    v_a = sqrt(v_g**2-2*v_g*(w_n*cos(chi)*cos(gamma)+w_e*sin(chi)*cos(gamma)-w_d*sin(gamma))+v_w**2) # Positive root    
    gamma_a = arcsin((v_g*sin(gamma) + w_d)/v_a) #WIND
    psi = chi - arcsin((-w_n*sin(chi) + w_e*cos(chi))/(v_a*cos(gamma_a))) #WIND    

#    wca=chi-psi    
    
#    # Wind fudge factor
#    var=np.array([2.26591188778e-12,
#    -2.58669180931e-10,
#    1.34535314401e-08,
#    -4.22108160844e-07,
#    8.91608591048e-06,
#    -0.000133990064321,
#    0.0014760589565,
#    -0.0121022332842,
#    0.0742186287642,
#    -0.339253887753,
#    1.14223842199,
#    -2.77208880132,
#    4.68057057378,
#    -5.18439733443,
#    3.37962414561,
#    0.0])
#    wind_param = v_w**15*var[0]+v_w**14*var[1]+v_w**13*var[2]+v_w**12*var[3]+v_w**11*var[4]+v_w**10*var[5]+v_w**9*var[6]+v_w**8*var[7]+v_w**7*var[8]+v_w**6*var[9]+v_w**5*var[10]+v_w**4*var[11]+v_w**3*var[12]+v_w**2*var[13]+v_w*var[14]+var[15]
    
#    phi=(phi_0+np.arctan((v_g/v_a)**2/np.cos(wca)*np.tan(phi_0))*(wind_param**v_w))#*.6#redefining phi for the wind correction angle
#    phi=np.arctan((v_g/v_a)**2/np.cos(wca)*np.tan(phi_0)) # Removed phi_0, possible bug?
    # Beard
#    W = np.arctan2(w_n,w_e)
#    phi = np.arctan(v_a**2/(g*3000)*np.cos(gamma)*np.cos(np.arcsin(v_w*cos(chi-W)/(v_a*cos(gamma)))))
    # New Phi
#    phi = np.arctan(v_g**2/(g*3000*np.sqrt(1-(1/v_a*(-w_n*sin(chi)+w_e*cos(chi)))**2)))
    phi = np.arctan(((w_n*cos(chi)+w_e*sin(chi))+np.sqrt((w_n*cos(chi)+w_e*sin(chi))**2-v_w**2+v_a**2))**2/(g*config['trajectory']['x']['max']*np.sqrt((v_a**2-(w_n*sin(chi)-w_e*cos(chi))**2-w_d**2)/(v_a**2-w_d**2))))
    
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
    E_batmax = m_battery*E_d*3.6/1000.0 # Max energy stored in battery (MJ)
    panel_efficiency = config['solar']['panel_efficiency'] # 0.25 # (FB)
    
    # Manipulated variables
#    Tp = Tp_0 #3.171 # 48.4 # Thrust (N) (function input)
#    alpha = alpha_0 # Angle of Attack (rad) (function input)
    rho = rho_11 * exp(-(g/(R_air*T_11))*(h-11000)) # Air density (kg/m**3)
    q = 1/2.0*rho*v_a**2 # Dynamic pressure (Pa)
    if(config['aircraft']['name']=='Aquila'):
        alpha = np.radians(m*g/(cos(phi)*q*S*0.0945)-0.6555/0.0945)
    elif(config['aircraft']['name'] == 'Aquila E216'):
        alpha = np.radians(m*g/(cos(phi)*q*S*0.095)-0.727/0.095)
    #phi = phi_0 #0.038 #2.059E-03 # 0.0001 # Bank Angle (rad) (function input)
    
    #### Atmospheric Effects
    #rho = rho_11 * exp(-(g/(R_air*T_11))*(h-11000)) # Air density (kg/m**3)
    
    # Pitch from AoA
    theta = gamma + alpha
    
    # Alpha in degress for fits
    alpha_deg = np.degrees(alpha)
    
    # CL from AoA, Lift slope line from xflr5
#    cl = (1.59-0.656)/(10-0)*(alpha*180/pi-0)+0.656
    if(config['aircraft']['name']=='Aquila'):
        cl = 0.0945*alpha_deg + 0.6555
    elif(config['aircraft']['name']=='Helios'):
        cl = 0.099*alpha_deg + 0.041
    elif(config['aircraft']['name'] == 'Aquila E216'):
        cl = 0.095*alpha_0*180/pi+0.727
    
    #### Drag Model
    ### Top Surface
    ## Top Reynolds Numbers
    # Flat plate Reynolds number
    Re = rho*v_a*chord/mu # Reynolds number
#    # Top surface laminar region
#    Re_Laminar_top = Re*xcrit_top
#    # Top surface transition region
#    theta_top = .671*xcrit_top/sqrt(Re_Laminar_top)
#    xeff_top = (27.78*theta_top*Re**(1/5.0))**(5/4.0)
#    Re_overlap_top = Re*xeff_top
#    # Top surface turbulent region
#    Re_Turbulent_top = Re*(1-xcrit_top+xeff_top)
#    
#    ## Top Skin Friction Drag Coefficient
#    # Laminar
#    c_f_laminar_top = ((1.328)/sqrt(Re_Laminar_top))
#    # Turbulent
#    c_f_turbulent_top = (.455)/((log10(Re_Turbulent_top))**(2.58))
#    # Transition
#    c_f_overlap_top = (.455)/((log10(Re_overlap_top))**(2.58))
#    # Total
#    c_f_smooth_top = (xcrit_top*c_f_laminar_top + (1-xcrit_top+xeff_top)*c_f_turbulent_top - xeff_top*c_f_overlap_top + c_f_laminar_top)
#    c_f_rough_top = c_f_smooth_top*roughness
#    # Parasitic drag coefficient 
#    C_D_p_top = k*c_f_rough_top*S_wet/S
#    
#    ### Bottom Surface
#    ## Reynolds Numbers
#    # Laminar
#    Re_Laminar_bottom = Re*xcrit_bottom
#    # Transition
#    theta_bottom = .671*xcrit_bottom/sqrt(Re_Laminar_bottom)
#    xeff_bottom = (27.78*theta_bottom*Re**(1/5.0))**(5/4.0)
#    Re_overlap_bottom = Re*xeff_bottom
#    # Turbulent
#    Re_Turbulent_bottom = Re*(1-xcrit_bottom+xeff_bottom)
#    
#    ## Bottom Skin Friction Drag Coefficient
#    # Laminar
#    c_f_laminar_bottom = ((1.328)/sqrt(Re_Laminar_bottom))
#    # Turbulent
#    c_f_turbulent_bottom = (.455)/((log10(Re_Turbulent_bottom))**(2.58))
#    # Transition
#    c_f_overlap_bottom = (.455)/((log10(Re_overlap_bottom))**(2.58))
#    # Total
#    c_f_smooth_bottom = c_f_laminar_bottom
#    c_f_rough_bottom = c_f_smooth_bottom*roughness
#    # Parasitic drag coefficient 
#    C_D_p_bottom = k*c_f_rough_bottom*S_wet/S
#    
#    ### Parasitic drag from gaps
#    C_D_gap = .00015*(cos(Lambda)**2)*(0.3*S)
#    
#    ### Parasitic drag coefficient
#    C_D_p_temp = C_D_p_top + C_D_p_bottom + C_D_gap
#    C_D_p = C_D_p_temp/(1-0.01)
    
    # New viscous/parasitic drag from xflr5
#    C_D_p = 6.2740486643e-07*alpha_deg**5 - 1.4412023268e-05*alpha_deg**4 + 1.1160259529e-04*alpha_deg**3 - 2.2563681473E-04*alpha_deg**2 - 8.6114793593E-05*alpha_deg + 1.0569281079E-02
    
    if(config['aircraft']['name'] == 'Aquila'):
        C_D_p = dsa*((dsd*alpha_deg+dsf)**dsb+(dsg*Re+dsh)**dsc)
    elif(config['aircraft']['name'] == 'Helios'):
        C_D_p = dsa * Re**(-1) + dsb * alpha_deg**(2) * Re**(-1)
    elif(config['aircraft']['name'] == 'Aquila E216'):
        C_D_p = (dsa + dsb*alpha_deg + dsc*Re + dsd*alpha_deg*Re) / (1.0 + dsf*alpha_deg + dsg*Re + dsh*alpha_deg*alpha_deg)
    
    
    # Oswald efficiency factor
    k_e = 0.4*C_D_p
    e_o = 1/((pi*AR*k_e) + (1/es))
    # Drag coefficient
    C_D = C_D_p + cl**2/(pi*AR*e_o)
    
    #### Flight Dynamics
    #q = 1/2.0*rho*v_a**2 # Dynamic pressure (Pa)
    
    L = q*cl*S # Lift (N) (simplified definition using q)
    D = C_D*q*S # Corrected Drag (N)
    
    nh = L*sin(phi)/(m*g) # Horizontal load factor
    nv = L*cos(phi)/(m*g) # Vertical load factor
    
    ### Propeller Max Theoretical Efficiency
    Adisk = pi * R_prop**2 # Area of disk
    e_prop = 2.0 / (1.0 + ( D / (Adisk * v_a**2.0 * rho/2.0) + 1.0 )**0.5)
    nu_prop = e_prop * e_motor
    
    # Thrust for constant va
    Tp =  D#(2*g*v_g*(sin(gamma) + D/(g*m)) - 2*g*(sin(gamma) + D/(g*m))*(w_n*cos(chi)*cos(gamma) - w_d*sin(gamma) + w_e*cos(gamma)*sin(chi)))/((2*v_g)/m - (2*w_n*cos(chi)*cos(gamma) - 2*w_d*sin(gamma) + 2*w_e*cos(gamma)*sin(chi))/m)
    
    #### Power
    P_N = P_payload + v_a*Tp/nu_prop # Power Needed by Aircraft
    
    # Solar
    solar_data = config['solar']['solar_data']
    lat = config['solar'][solar_data]['latitude'] # 35.0853
    lon = config['solar'][solar_data]['longitude'] # -106.6056
    elevation = config['solar'][solar_data]['elevation'] # 1.619
    altitude = config['solar'][solar_data]['altitude'] # 20
    year = config['solar'][solar_data]['year'] # 2016
    month = config['solar'][solar_data]['month'] # 12
    day = config['solar'][solar_data]['day'] # 21
    zone = config['solar'][solar_data]['zone'] # -7
    
#    Tp = Tp_0 + 12*(34.9269232954023 - v_a)
    
    if(mode==1 or mode==5):
        solar_data = solarFlux(smartsData,lat,lon,elevation, altitude, year, month, day, t/3600.0, zone, orientation=True,
                               phi = phi,theta = theta, psi = psi)
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
    # Solar Efficiency - updated (fit to -50 deg C)
    eta = config['solar']['panel_efficiency_function']['eta'] # 0.12
    beta = config['solar']['panel_efficiency_function']['beta'] # 0.0021888986107182 # 0.002720315 (old)
    Tref = config['solar']['panel_efficiency_function']['Tref'] # 298.15
    gamma_s = config['solar']['panel_efficiency_function']['gamma_s'] # 0.413220518404272 # 0.513153469 (old)
    T_noct = config['solar']['panel_efficiency_function']['T_noct'] # 20.0310337470507 # 20.0457889 (old)
    G_noct = config['solar']['panel_efficiency_function']['G_noct'] # 0.519455027587048 # 0.527822252 (old)
    if(config['aircraft']['name'] == 'Aquila' or config['aircraft']['name'] == 'Aquila E216'):
        panel_efficiency = eta*(1-beta*(T_11-Tref+(T_noct-20)*G_sol/G_noct)+gamma_s*log10(G_sol+0.01))
    P_solar = G_sol*S*panel_efficiency

    # Flight Dynamics - with WIND
#    dv_g_dt = ((Tp-D)/(m*g)-sin(gamma))*g
    dv_a_dt = 0 
    dgamma_dt = g/v_g*(nv-cos(gamma))
    dchi_dt = g/v_g*(nh/cos(gamma))*cos(chi-psi)
    dh_dt = v_g*sin(gamma)
    dx_dt = v_g*cos(chi)*cos(gamma)
    dy_dt = v_g*sin(chi)*cos(gamma)
    dist = (sqrt(x**2+y**2))
    radius = v_g**2/(g*tan(phi))# Flight path radius
    
    radius_max = config['trajectory']['x']['max'] # 3000 (m)
    
    # Calculate sideslip
    V_bar_g = np.array([[dx_dt],[dy_dt]])
    V_bar_w = np.array([[w_n],[w_e]])
    V_bar_a = V_bar_g - V_bar_w
    V_a_angle = np.arctan2(V_bar_a[1],V_bar_a[0])
    if(V_a_angle<0):
        V_a_angle = V_a_angle + 2*np.pi
    beta = V_a_angle - np.mod(psi,2*np.pi)
    
    # Power
    P_bat = P_solar - P_N # Power used to charge or discharge battery (W)
    dE_Batt_dt = P_bat*1e-6 # (Convert to MJ)
    h_0 = config['trajectory']['h']['initial_value']
    TE = E_batt + m*g*(h-h_0)*1e-6
    
    # Clip d_E_batt at maximum battery
    bat_switch = 1/(1+exp(-10*(E_batmax - E_batt)))
    if(dE_Batt_dt>0):
        dE_Batt_dt = dE_Batt_dt * bat_switch
    
    # Collect model outputs into dictionary
    model_output = {"dv_a_dt":dv_a_dt,
                    "dgamma_dt":dgamma_dt,
                    "dchi_dt":dchi_dt,
                     "dh_dt":dh_dt,
                     "dx_dt":dx_dt,
                     "dy_dt":dy_dt,
                     "dE_Batt_dt":dE_Batt_dt,
                     "radius_const":radius-radius_max,
                     "P_N":P_N,
                     'cl':cl,
                     'mu':mu_solar,
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
                     'c_d_p':C_D_p,
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
                     'psi':psi,
                     'gamma_a':gamma_a,
                     'v_a':v_a,
                     'phi':phi,
                     'tp':Tp,
                     'v_g':v_g_actual,
                     'beta':beta,
                     'v_a_angle':V_a_angle,
                     'alpha':alpha}
    
    return model_output