# -*- coding: utf-8 -*-
from gekko import GEKKO
import datetime
import numpy as np
from atmosphere import getAtmosphere

def init_model(config):

    # Select server
    server = config['optimization']['server']
    
    # Application name
    app = 'hale_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    
    #Initialize model
    m = GEKKO(server,app)
    
    #%% Model


    #%% Constants
    
    pi = 3.141592653
    # Atmospheric Effects
    g = 9.80665 # Gravity (m/s^2)
    R_air = 287.041 # Gas Constant for air (m^2/(s^2 K))
    rho_11 = 0.364 # Standard air density at 11 km (kg/m^3)
    T_11 = 216.66 # Standard air temp at 11 km (K)

    # Flight Dynamics
    mass = config['aircraft']['mass_total']['value'] # Total Mass (kg) (FB)
    E_d = config['aircraft']['battery']['energy_density']['value'] # Battery energy density (W*hr/kg) (FB)
    m_battery = config['aircraft']['battery']['mass']['value'] # (kg) (FB)
    max_dist = config['trajectory']['x']['max'] # Maximum orbit radius (m)
    
    # Drag Model (Translated from Judd's Matlab code)
    AR = config['aircraft']['aspect_ratio'] # Aspect ratio (FB)
    S = config['aircraft']['wing_top_surface_area']['value'] # Wing and solar panel area (m^2) (FB)
    b = np.sqrt(S*AR) # Wingspan (m)
    chord = b/AR # Chord (m)
    es = config['aircraft']['inviscid_span_efficiency'] # Inviscid span efficiency (Judd)
    
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
    
    # Propeller Efficiency
    R_prop = config['aircraft']['propeller_radius']['value'] # Propeller Radius (m) - Kevin
    
    # Power
    e_motor = config['aircraft']['motor_efficiency'] # Efficiency of motor
    P_payload = config['aircraft']['power_for_payload']['value'] # Power for payload (W)
    E_batmax = m_battery*E_d*3.6/1000 #m.Const(m_battery.value*E_d.value*3.6/1000,'E_batmax') # Max energy stored in battery (MJ)
    use_solar_orientation = 1 # Switch between horizontal and oriented solar
    
    # Solar Efficiency - updated (fit to -50 deg C)
    eta_s = config['solar']['panel_efficiency_function']['eta']
    beta_s = config['solar']['panel_efficiency_function']['beta'] # 0.002720315 (old)
    Tref = config['solar']['panel_efficiency_function']['Tref']
    gamma_s = config['solar']['panel_efficiency_function']['gamma_s'] # 0.513153469 (old)
    T_noct = config['solar']['panel_efficiency_function']['T_noct'] # 20.0457889 (old)
    G_noct = config['solar']['panel_efficiency_function']['G_noct'] # 0.527822252 (old)
    
    # Initial Conditions (from final initialization)
    h_0 = config['trajectory']['h']['initial_value']#m.Const(18288,'h_0')
    if(config['wind']['use_wind']):
        v_a0 = config['trajectory']['v']['initial_value']#m.Const(35,'v_0') #28.4
        v_g0 = config['trajectory']['v']['initial_value']
    else:
        v_0 = config['trajectory']['v']['initial_value']#m.Const(35,'v_0') #28.4
    gamma_0 = config['trajectory']['gamma']['initial_value']#m.Const(0,'gamma_0') # Initial flight path angle (rad)
    alpha_0 = config['trajectory']['alpha']['initial_value']#m.Const(0.0874,'alpha_0') # Initial angle of attack (rad) gives cl of 1.05
    psi_0 = config['trajectory']['psi']['initial_value']#m.Const(0,'psi_0') # Initial heading (rad)
    phi_0 = config['trajectory']['phi']['initial_value']#m.Const(0.0656,'phi_0') # Initial bank angle (rad)
    tp_0 = config['trajectory']['tp']['initial_value']#m.Const(110.54,'tp_0') # Initial thrust
    SOC_initial = config['aircraft']['battery']['initial_state_of_charge']
    E_Batt_0 = E_batmax*SOC_initial#m.Const(E_batmax.value*SOC_initial.value,'E_Batt_0') # Initial Battery Charge
    
    #%% End Constants
    
    
    
    #%% Parameters
    
    t = m.Param(name='t',value=0) # Time
    flux = m.Param(name='flux',value=0) # Direct tracking flux from SMARTS
    sunset = m.Param(name='sunset',value=0) # Time step at which sun sets
    zenith = m.Param(name='zenith',value=0) # Solar zenith from SMARTS (0=up, 90=horizon)
    azimuth = m.Param(name='azimuth',value=0) # Solar azimuth from SMARTS (clockwise from north)
    sn1 = m.Param(name='sn1',value=0) # Normalized sun direction vector
    sn2 = m.Param(name='sn2',value=0) # Normalized sun direction vector
    sn3 = m.Param(name='sn3',value=0) # Normalized sun direction vector
    sun_h = m.Param(name='sun_h',value=0) # Flux on horizontal surface
    
    # FVs and MVs need to be parameters
    tp = m.MV(name='tp',value=tp_0) #3.171 
    alpha = m.MV(name='alpha',value=alpha_0) # Angle of Attack (rad)
    phi = m.MV(name='phi',value=phi_0) #0.038 
    p_bat = m.MV(name='p_bat',value=0) # Power used to charge or discharge battery (W)
    
    if(config['wind']['use_wind']):
        # Wind
        w_n = m.Param(name='w_n',value=config['wind']['w_n'])
        w_e = m.Param(name='w_e',value=config['wind']['w_e'])
        w_d = m.Param(name='w_d',value=config['wind']['w_d'])
    #%% End Parameters
    
    
    
    #%% Variables
    
    # Atmospheric Effects
    h = m.CV(name='h',value=h_0,lb=config['trajectory']['h']['min']*0.98,ub=config['trajectory']['h']['max']*1.02) # Height from sea level (m) (27432 m = 90,000 ft, 18288 m = 60,000 ft)
    
    
#    # Prepare atmosphere data for spline
#    hlist = np.linspace(15000,30000)
#    rhoData = np.zeros(len(hlist))
#    T_airData = np.zeros(len(hlist))
#    muData = np.zeros(len(hlist))
#    for i,hval in enumerate(hlist):
#        rho,pressure,T_air,mu,kinematicViscosity = getAtmosphere(hval)
#        rhoData[i] = rho
#        T_airData[i] = T_air
#        muData[i] = mu
#    # Create splines
#    rho_s = m.Var(name='rho')
#    T_air_s = m.Var(name='T_air')
#    mu_s = m.Var(name='mu')
#    m.cspline(h,rho_s,hlist,rhoData,False)
#    m.cspline(h,T_air_s,hlist,T_airData,False)
#    m.cspline(h,mu_s,hlist,muData,False)
    
    # Flight Dynamics
    x = m.Var(name='x',value=0) # Horizontal distance (m)
    y = m.Var(name='y',value=-3000) # Other horizontal distance
    if(config['wind']['use_wind']):
        v_a = m.Var(name='v_a',value=v_a0,lb=0) # Velocity (m/s)
        v_g = m.Var(name='v_g',value=v_g0,lb=0) # Velocity (m/s)
        chi = m.Var(name='chi',value=psi_0) # Heading angle
        h_a = m.Var(value=h.value,name='h_a')
        x_a = m.Var(value=x.value,name='x_a')
        y_a = m.Var(value=y.value,name='y_a')
    else:
        v = m.Var(name='v',value=v_0,lb=0) # Velocity (m/s)
        psi = m.Var(name='psi',value=psi_0) # Heading angle
    gamma = m.Var(name='gamma',value=gamma_0,lb=-0.085,ub=0.085) # Flight path angle (rad) - previously >= -1 <= 1
    
    dist = m.CV(name='dist',value=m.sqrt(x**2+y**2),lb=0,ub=max_dist*1.1)
#    dist = m.CV(name='dist',value=x**2/(3000**2)+y**2/(30000**2),lb=0,ub=1.0001)
    if(config['aircraft']['name'] == 'Aquila'):
        cl = m.CV(name='cl',value=0.0945*alpha_0*180/pi+0.6555,ub=1.55)
    elif(config['aircraft']['name'] == 'Helios'):
        cl = m.CV(name='cl',value=0.099*alpha_0*180/pi+0.041,ub=1.55)
    elif(config['aircraft']['name'] == 'Aquila E216'):
        cl = m.CV(name='cl',value=0.095*alpha_0*180/pi+0.727,ub=1.55)
    
    # Solar
    mu_clipped = m.Var(name='mu_clipped',value=0,lb=0)
    mu_slack = m.Var(name='mu_slack',value=0,lb=0)
    
    # Power
    e_batt = m.Var(name='e_batt',value=E_Batt_0,ub=E_batmax) # Energy stored in battery (MJ)
    te = m.Var(name='te',value=E_Batt_0+mass*g*(h-h_0)*1e-6) # Total energy (MJ)
    p_total = m.Var(name='p_total',value=0,lb=0)
#    if(config['aircraft']['name'] == 'Aquila' or config['aircraft']['name'] == 'Aquila E216'):
#        P_N = m.Var(name='P_N',0,ub=4000*4)
#    elif(config['aircraft']['name'] == 'Helios'):
#        P_N = m.Var(name='P_N',0,ub=1500*14)
    workD = m.Var(value=0,name='workD')
    workTp = m.Var(value=0,name='workTp')
    if(config['wind']['use_wind']):
        KE = m.Var(value=0.5*mass*v_g**2,name='KE')
    else:
        KE = m.Var(value=0,name='KE')
    PE = m.Var(value=0,name='PE')
    E_balance = m.Var(value=KE+PE+workD+workTp,name='E_balance')
    
    
    #%% End Variables
    
    
    
    #%% Intermediates
    
    #### Atmospheric Effects
    rho = m.Intermediate(rho_11*m.exp(-(g/(R_air*T_11))*(h-11000)),'rho')
    T_air = m.Intermediate(-56.46+273.15,'T_air')
    mu = m.Intermediate(1.458e-6*m.sqrt(T_air)/(1+110.4/T_air),'mu') # Dynamic viscosity of air (kg/(m s))
#    rho = rho_s
#    T_air = T_air_s
#    mu = mu_s
    
    
    # Pitch from AoA
    theta = m.Intermediate(gamma+alpha,'theta')
    
    hdot = m.Intermediate(h.dt(),'hdot')
    
    if(config['wind']['use_wind']):
        v_w = m.Intermediate(m.sqrt(w_n**2+w_e**2+w_d**2),'v_w')
        gamma_a = m.Intermediate((v_g*gamma + w_d)/v_a,'gamma_a')
        psi = m.Intermediate(chi - m.asin((-w_n*m.sin(chi) + w_e*m.cos(chi))/(v_a*m.cos(gamma_a))),'psi')
        dx = m.Intermediate(x.dt(),'dx')
        dy = m.Intermediate(y.dt(),'dy')
        psi_n = m.Intermediate(m.cos(psi),'psi_n')
        psi_e = m.Intermediate(m.sin(psi),'psi_e')
        v_a_n = m.Intermediate(dx-w_n,'v_a_n')
        v_a_e = m.Intermediate(dy-w_e,'v_a_e')
        beta = m.Intermediate(m.acos((v_a_n * psi_n + v_a_e * psi_e)/(m.sqrt(v_a_n**2+v_a_e**2)*m.sqrt(psi_n**2+psi_e**2))),'beta')
    
    #### Drag Model
    ### Top Surface
    ## Top Reynolds Numbers
    # Flat plate Reynolds number
    if(config['wind']['use_wind']):
        Re = m.Intermediate(rho*v_a*chord/mu,'Re') # Reynolds number
    else:
        Re = m.Intermediate(rho*v*chord/mu,'Re') # Reynolds number
    
    ### Parasitic drag from xflr5
    alpha_deg = m.Intermediate(alpha*180/pi,'alpha_deg')
    if(config['aircraft']['name'] == 'Aquila'):
        C_D_p = m.Intermediate(dsa*((dsd*alpha_deg+dsf)**dsb+(dsg*Re+dsh)**dsc),'C_D_p')
    elif(config['aircraft']['name'] == 'Helios'):
        C_D_p = m.Intermediate(dsa * Re**(-1) + dsb * alpha_deg**(2) * Re**(-1),'C_D_p')
    elif(config['aircraft']['name'] == 'Aquila E216'):
        C_D_p = m.Intermediate((dsa + dsb*alpha_deg + dsc*Re + dsd*alpha_deg*Re) / (1.0 + dsf*alpha_deg + dsg*Re + dsh*alpha_deg*alpha_deg),'C_D_p')
    
    # Oswald efficiency factor
    k_e = m.Intermediate(0.4*C_D_p,'k_e')
    e_o = m.Intermediate(1/((pi*AR*k_e)+(1/es)),'e_o')
    # Drag coefficient
    C_D = m.Intermediate(C_D_p+cl**2/(pi*AR*e_o),'C_D')
#    cd = m.Intermediate(C_D,'cd') # Added for data handling
    
    #### Flight Dynamics
    if(config['wind']['use_wind']):
        q = m.Intermediate(1/2*rho*v_a**2,'q') # Dynamic pressure (Pa)
    else:
        q = m.Intermediate(1/2*rho*v**2,'q') # Dynamic pressure (Pa)
    
    L = m.Intermediate(q*cl*S,'L') # Lift (N) (simplified definition using q)
    D = m.Intermediate(C_D*q*S,'D') # Drag (N) (Nathaniel correction - use C_D, as C_D_p is already included)
    
    nh = m.Intermediate(L*m.sin(phi)/(mass*g),'nh') # Horizontal load factor
    nv = m.Intermediate(L*m.cos(phi)/(mass*g),'nv') # Vertical load factor
    
    ### Propeller Max Theoretical Efficiency
    Adisk = m.Intermediate(pi*R_prop**2,'Adisk') # Area of disk
    if(config['wind']['use_wind']):
        e_prop = m.Intermediate(2.0/(1.0+(D/(Adisk*v_a**2.0*rho/2.0)+1.0)**0.5),'e_prop')
    else:
        e_prop = m.Intermediate(2.0/(1.0+(D/(Adisk*v**2.0*rho/2.0)+1.0)**0.5),'e_prop')
    nu_prop = m.Intermediate(e_prop*e_motor,'nu_prop')
    
    ### Power
    if(config['wind']['use_wind']):
        P_N = m.Intermediate(P_payload+v_a*tp/nu_prop,'P_N') # Power Needed by Aircraft
    else:
        P_N = m.Intermediate(P_payload+v*tp/nu_prop,'P_N') # Power Needed by Aircraft
    
    #### Solar (with orientation correction)
    c1 = m.Intermediate(m.cos(-phi),'c1')
    c2 = m.Intermediate(m.cos(-theta),'c2')
    c3 = m.Intermediate(m.cos(psi),'c3')
    s1 = m.Intermediate(m.sin(-phi),'s1')
    s2 = m.Intermediate(m.sin(-theta),'s2')
    s3 = m.Intermediate(m.sin(psi),'s3')
    n1 = m.Intermediate(c1*s2*s3-c3*s1,'n1')
    n2 = m.Intermediate(c1*c3*s2+s1*s3,'n2')
    n3 = m.Intermediate(c1*c2,'n3')
    nn = m.Intermediate(m.sqrt(n1**2+n2**2+n3**2),'nn')
    mu_solar = m.Intermediate(sn1*n1/nn+sn2*n2/nn+sn3*n3/nn,'mu_solar')
    G_sol = m.Intermediate(flux*mu_clipped,'G_sol') # Orientation adjusted solar flux (W/m^2)
    if(config['aircraft']['name'] == 'Aquila' or config['aircraft']['name'] == 'Aquila E216'):
        panel_efficiency = m.Intermediate(eta_s*(1-beta_s*(T_11-Tref+(T_noct-20)*G_sol/G_noct)+gamma_s*m.log10(G_sol+0.01)),'panel_efficiency') # Solar panel efficiency (Nathaniel)
    elif(config['aircraft']['name'] == 'Helios'):
        panel_efficiency = m.Intermediate(config['solar']['panel_efficiency'],'panel_efficiency')
    solar_o = m.Intermediate(mu_clipped*panel_efficiency*S*flux,'solar_o') # Total power generated by panel (W)
    solar_h = m.Intermediate(sun_h*panel_efficiency*S,'solar_h')
    P_solar = m.Intermediate(solar_o*use_solar_orientation+solar_h*(1-use_solar_orientation),'P_solar') # Allows switching solar orientation
    
    #%% End Intermediates
    
    
    
    #%% Equations
    
    # Flight Dynamics
    if(config['wind']['use_wind']):
        m.Equation(v_g.dt()==((tp-D)/(mass*g)-m.sin(gamma))*g)
        m.Equation(gamma.dt()*v_g==g*(nv-m.cos(gamma)))
        m.Equation(chi.dt()*v_g*m.cos(gamma)==g*nh*m.cos(chi-psi))
        m.Equation(h.dt()==v_g*m.sin(gamma))
        m.Equation(x.dt()==v_g*m.cos(chi)*m.cos(gamma))
        m.Equation(y.dt()==v_g*m.sin(chi)*m.cos(gamma))
        m.Equation(h_a.dt()==v_a*m.sin(gamma_a))
        m.Equation(x_a.dt()==v_a*m.cos(psi)*m.cos(gamma_a))
        m.Equation(y_a.dt()==v_a*m.sin(psi)*m.cos(gamma_a))
        m.Equation(dist==m.sqrt(x**2+y**2))
#        m.Equation(dist==x**2/(3000**2)+y**2/(30000**2))
        m.Equation(v_a == m.sqrt(v_g**2-2*v_g*(w_n*m.cos(chi)*m.cos(gamma)+w_e*m.sin(chi)*m.cos(gamma)-w_d*m.sin(gamma))+v_w**2))
    else:
        m.Equation(v.dt()==((tp-D)/(mass*g)-m.sin(gamma))*g)
        m.Equation(gamma.dt()*v==g*(nv-m.cos(gamma)))
        m.Equation(psi.dt()*v*m.cos(gamma)==g*nh)
        m.Equation(h.dt()==v*m.sin(gamma))
        m.Equation(x.dt()==v*m.cos(psi)*m.cos(gamma))
        m.Equation(y.dt()==v*m.sin(psi)*m.cos(gamma))
        m.Equation(dist==m.sqrt(x**2+y**2))
#        m.Equation(dist==x**2/(12000**2)+y**2/(3000**2))
    
    # Power
    m.Equation(e_batt.dt()==p_bat*1e-6) # (Convert to MJ) p_bat is the charging rate of the battery.
    m.Equation(p_total==P_solar-p_bat-P_N)
    m.Equation(te==e_batt+mass*g*(h-h_0)*1e-6)
    
    # Solar
    m.Equation(mu_clipped==mu_solar+mu_slack)
    m.Equation(mu_clipped*mu_slack<=1e-4)
    
    # Energy Balance
    if(config['wind']['use_wind']):
        m.Equation(workD.dt()==D*v_g)
        m.Equation(workTp.dt()==tp*v_g)
        m.Equation(KE.dt()==v_g*mass*v_g.dt())
        m.Equation(PE.dt()==mass*g*h.dt())
    else:
        m.Equation(workD.dt()==D*v)
        m.Equation(workTp.dt()==tp*v)
        m.Equation(KE.dt()==v*mass*v.dt())
        m.Equation(PE.dt()==mass*g*h.dt())
    m.Equation(E_balance == KE+PE+workD+workTp)
    

    # Constraints
    if(config['aircraft']['name'] == 'Aquila'):
        m.Equation(cl==0.0945*alpha*180/pi+0.6555)
    elif(config['aircraft']['name'] == 'Helios'):
        m.Equation(cl==0.099*alpha*180/pi+0.041)
    elif(config['aircraft']['name'] == 'Aquila E216'):
        m.Equation(cl==0.095*alpha_0*180/pi+0.727)
        
#    if(config['wind']['use_wind']):
#        m.Equation(P_N == P_payload+v_a*tp/nu_prop)
#    elif(config['aircraft']['name'] == 'Helios'):
#        m.Equation(P_N == P_payload+v*tp/nu_prop)
    
    # Objective
    m.Obj(-te)
    
    # Weight with previous solution
    m.x_prev = m.Param()
    m.y_prev = m.Param()
    m.h_prev = m.Param()
    m.gamma_prev = m.Param()
    m.convex_weight = m.Param()
    m.convex_horizon = m.Param()
    
    m.Obj((x-m.x_prev)**2*m.convex_weight*m.convex_horizon)
    m.Obj((y-m.y_prev)**2*m.convex_weight*m.convex_horizon)
    m.Obj((h-m.h_prev)**2*m.convex_weight*m.convex_horizon)
    m.Obj((gamma-m.gamma_prev)**2*m.convex_weight*m.convex_horizon)
    
    
    #%% End Equations
    
    
    #%% End Model
    
    m.t=t
    m.flux=flux
    m.sunset=sunset
    m.zenith=zenith
    m.azimuth=azimuth
    m.sn1=sn1
    m.sn2=sn2
    m.sn3=sn3
    m.sun_h=sun_h
    m.tp=tp
    m.alpha=alpha
    m.phi=phi
    m.p_bat=p_bat
    m.h=h
    m.gamma=gamma
    m.psi=psi
    m.x=x
    m.y=y
    m.dist=dist
    m.cl=cl
    m.mu_clipped=mu_clipped
    m.mu_slack=mu_slack
    m.e_batt=e_batt
    m.te=te
    m.p_total=p_total
    if(config['wind']['use_wind']):
        m.chi = chi
        m.v_a = v_a
        m.v_g = v_g
        m.x_a = x_a
        m.y_a = y_a
        m.h_a = h_a
        m.w_n = w_n
        m.w_e = w_e
        m.w_d = w_d
    else:
        m.v=v
    
    return m