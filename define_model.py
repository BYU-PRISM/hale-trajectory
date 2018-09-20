# -*- coding: utf-8 -*-
from gekko import GEKKO
import datetime

def define_model(config):

    # Select server
    server = config.server
    
    # Application name
    app = 'hale_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    
    #Initialize model
    m = GEKKO(server,app)

    #%% Constants
    
    pi = 3.141592653
    # Atmospheric Effects
    g = 9.80665 # Gravity (m/s^2)
    R_air = 287.041 # Gas Constant for air (m^2/(s^2 K))
    rho_11 = 0.364 # Standard air density at 11 km (kg/m^3)
    T_11 = 216.66 # Standard air temp at 11 km (K)

    # Flight Dynamics
    mass = config.aircraft.mass_total.value # Total Mass (kg) 
    E_d = config.aircraft.battery_energy_density.value # Battery energy density (W*hr/kg) 
    m_battery = config.aircraft.mass_battery.value # (kg) 
    max_dist = config.distance.max # Maximum orbit radius (m)
    
    # Drag Model (Translated from Judd's Matlab code)
    S = config.aircraft.wing_top_surface_area.value # Wing and solar panel area (m^2) 
    chord = config.aircraft.chord.value
    
    # Propeller Efficiency
    R_prop = config.aircraft.propeller_radius.value # Propeller Radius (m) - Kevin
    
    # Power
    e_motor = config.aircraft.motor_efficiency.value # Efficiency of motor
    P_payload = config.aircraft.power_for_payload.value # Power for payload (W)
    E_batmax = m_battery*E_d*3.6/1000 #m.Const(m_battery.value*E_d.value*3.6/1000,'E_batmax') # Max energy stored in battery (MJ)
    
    # Initial Conditions (from final initialization)
    h_0 = config.h.initial_value
    if(config.use_wind):
        v_a0 = config.aircraft.v.initial_value
        v_g0 = config.aircraft.v.initial_value
    else:
        v_0 = config.aircraft.v.initial_value
    gamma_0 = config.aircraft.gamma.level # Initial flight path angle (rad)
    alpha_0 = config.aircraft.alpha.initial_value # Initial angle of attack (rad) 
    psi_0 = config.aircraft.psi.initial_value # Initial heading (rad)
    phi_0 = config.aircraft.phi.initial_value # Initial bank angle (rad)
    tp_0 = config.aircraft.tp.initial_value # Initial thrust
    SOC_initial = config.aircraft.battery_initial_SOC.value # Initial state of charge
    E_Batt_0 = E_batmax*SOC_initial # Initial Battery Charge
    
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
    
    if(config.use_wind):
        # Wind
        w_n = m.Param(name='w_n',value=config.w_n)
        w_e = m.Param(name='w_e',value=config.w_e)
        w_d = m.Param(name='w_d',value=config.w_d)
    
    #%% Variables
    
    # Atmospheric Effects
    h = m.Var(name='h',value=h_0,lb=config.h.min,ub=config.h.max) # Height from sea level (m) (27432 m = 90,000 ft, 18288 m = 60,000 ft)
    
    # Flight Dynamics
    if(config.use_wind):
        v_a = m.Var(name='v_a',value=v_a0,lb=config.aircraft.v.min) # Velocity (m/s)
        v_g = m.Var(name='v_g',value=v_g0) # Velocity (m/s)
        chi = m.Var(name='chi',value=psi_0) # Heading angle
    else:
        v = m.Var(name='v',value=v_0,lb=config.aircraft.v.min) # Velocity (m/s)
        psi = m.Var(name='psi',value=psi_0) # Heading angle
    gamma = m.Var(name='gamma',value=gamma_0,lb=config.aircraft.gamma.min,ub=config.aircraft.gamma.max) # Flight path angle (rad)
    x = m.Var(name='x',value=config.x.initial_value) # Horizontal distance (m)
    y = m.Var(name='y',value=config.y.initial_value) # Other horizontal distance
    dist = m.CV(name='dist',value=m.sqrt(x**2+y**2),lb=0,ub=max_dist*1.1)
    
    # Solar
    mu_clipped = m.Var(name='mu_clipped',value=0,lb=0)
    mu_slack = m.Var(name='mu_slack',value=0,lb=0)
    
    # Power
    e_batt = m.Var(name='e_batt',value=E_Batt_0,ub=E_batmax) # Energy stored in battery (MJ)
    te = m.Var(name='te',value=E_Batt_0+mass*g*(h-h_0)*1e-6) # Total energy (MJ)
    p_total = m.Var(name='p_total',value=0,lb=0) # Energy balance
    
    
    #%% Intermediates
    
    #### Atmospheric Effects
    rho = m.Intermediate(rho_11*m.exp(-(g/(R_air*T_11))*(h-11000)),'rho') # Air density
    T_air = m.Intermediate(-56.46+273.15,'T_air') # Air temperature (isothermal region)
    mu = m.Intermediate(1.458e-6*m.sqrt(T_air)/(1+110.4/T_air),'mu') # Dynamic viscosity of air (kg/(m s))
    
    # Pitch from AoA
    theta = m.Intermediate(gamma+alpha,'theta')
    
    if(config.use_wind):
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
    
    # Flat plate Reynolds number
    if(config.use_wind):
        Re = m.Intermediate(rho*v_a*chord/mu,'Re') # Reynolds number
    else:
        Re = m.Intermediate(rho*v*chord/mu,'Re') # Reynolds number
        
    # Get alpha in degrees for fitted functions
    alpha_deg = alpha*180/pi
    
    # Wing lift coefficient
    cl = m.Intermediate(config.aircraft.CL(alpha_deg,Re),'cl')
    
    # Wing drag coefficient   
    C_D = m.Intermediate(config.aircraft.CD(alpha_deg,Re),'C_D')
    
    #### Flight Dynamics
    if(config.use_wind):
        q = m.Intermediate(1/2*rho*v_a**2,'q') # Dynamic pressure (Pa)
    else:
        q = m.Intermediate(1/2*rho*v**2,'q') # Dynamic pressure (Pa)
    
    L = m.Intermediate(q*cl*S,'L') # Lift (N)
    D = m.Intermediate(C_D*q*S,'D') # Drag (N)
    
    nh = m.Intermediate(L*m.sin(phi)/(mass*g),'nh') # Horizontal load factor
    nv = m.Intermediate(L*m.cos(phi)/(mass*g),'nv') # Vertical load factor
    
    ### Propeller Max Theoretical Efficiency
    Adisk = m.Intermediate(pi*R_prop**2,'Adisk') # Area of disk
    if(config.use_wind):
        e_prop = m.Intermediate(2.0/(1.0+(tp/(Adisk*v_a**2.0*rho/2.0)+1.0)**0.5),'e_prop')
    else:
        e_prop = m.Intermediate(2.0/(1.0+(tp/(Adisk*v**2.0*rho/2.0)+1.0)**0.5),'e_prop')
    nu_prop = m.Intermediate(e_prop*e_motor,'nu_prop')
    
    ### Power
    if(config.use_wind):
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
    mu_solar = m.Intermediate(sn1*n1/nn+sn2*n2/nn+sn3*n3/nn,'mu_solar') # Obliquity factor
    G_sol = m.Intermediate(flux*mu_clipped,'G_sol') # Orientation adjusted solar flux (W/m^2)
    panel_efficiency = m.Intermediate(config.aircraft.panel_efficiency(G_sol,m),'panel_efficiency') # Solar panel efficiency
    P_solar = m.Intermediate(mu_clipped*panel_efficiency*S*flux,'P_solar') # Total power generated by panel (W)
    
    #%% Equations
    
    # Flight Dynamics
    if(config.use_wind):
        m.Equation(v_g.dt()==((tp-D)/(mass*g)-m.sin(gamma))*g)
        m.Equation(gamma.dt()*v_g==g*(nv-m.cos(gamma)))
        m.Equation(chi.dt()*v_g*m.cos(gamma)==g*nh*m.cos(chi-psi))
        m.Equation(h.dt()==v_g*m.sin(gamma))
        m.Equation(x.dt()==v_g*m.cos(chi)*m.cos(gamma))
        m.Equation(y.dt()==v_g*m.sin(chi)*m.cos(gamma))
        m.Equation(dist==m.sqrt(x**2+y**2))
        m.Equation(v_a == m.sqrt(v_g**2-2*v_g*(w_n*m.cos(chi)*m.cos(gamma)+w_e*m.sin(chi)*m.cos(gamma)-w_d*m.sin(gamma))+v_w**2))
    else:
        m.Equation(v.dt()==((tp-D)/(mass*g)-m.sin(gamma))*g)
        m.Equation(gamma.dt()*v==g*(nv-m.cos(gamma)))
        m.Equation(psi.dt()*v*m.cos(gamma)==g*nh)
        m.Equation(h.dt()==v*m.sin(gamma))
        m.Equation(x.dt()==v*m.cos(psi)*m.cos(gamma))
        m.Equation(y.dt()==v*m.sin(psi)*m.cos(gamma))
        m.Equation(dist==m.sqrt(x**2+y**2))
    
    # Power
    m.Equation(e_batt.dt()==p_bat*1e-6) # (Convert to MJ) p_bat is the charging rate of the battery.
    m.Equation(p_total==P_solar-p_bat-P_N)
    m.Equation(te==e_batt+mass*g*(h-h_0)*1e-6)
    
    # Solar
    # This clips mu to be greater than 0
    m.Equation(mu_clipped==mu_solar+mu_slack)
    m.Equation(mu_clipped*mu_slack<=1e-4)
    
    # Objective - Maximize total energy
    m.Obj(-te)
    
    
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
    if(config.use_wind):
        m.chi = chi
        m.v_a = v_a
        m.v_g = v_g
        m.beta = beta
    else:
        m.v=v
    
    return m