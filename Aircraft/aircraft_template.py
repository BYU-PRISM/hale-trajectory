
import numpy as np
import numbers
from utilities import Param, Var
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Aircraft_Template:
    
    def __init__(self):
        # Battery
        self.mass_battery = Param(136.7,'kg')
        self.battery_energy_density = Param(350,'Whr/kg')
        self.battery_initial_SOC = Param(0.20,'N/A') # state of charge
        self.battery_max = Param(self.mass_battery.value*self.battery_energy_density.value*0.0036,'MJ')
        # Mass
        self.mass_payload = Param(25,'kg')
        self.mass_structure = Param(213,'kg')
        self.mass_total = Param(self.mass_structure.value+self.mass_battery.value
                                +self.mass_payload.value,'kg')
        # Wing
        self.chord = Param(1.41,'m')
        self.wing_top_surface_area = Param(60,'m^2')
        # Propeller
        self.propeller_radius = Param(2,'m')
        # Power
        self.motor_efficiency = Param(0.95,'N/A')
        self.power_for_payload = Param(250,'W')
        # Dynamics
        self.v = Var(
                    ss_initial_guess = 33,
                    min = 0,
                    units = 'm/s',
                    description = 'velocity'
                     )
        self.gamma = Var( 
                    max = float(np.radians(5)),
                    min = float(np.radians(-5)),
                    up = 0.02,
                    level = 0,
                    down = -0.02,
                    mode = 'level',
                    units = 'radians',
                    description = 'flight path angle (pitch)'
                    )
        self.psi = Var(
                    # Initial value will be set based on clockwise/counterclockwise
                    units = 'radians',
                    description = 'heading angle (yaw)'
                    )
        # Controls
        self.tp = Var(
                    max = 500,
                    min = 0.01,
                    dmax = 50,
                    dcost = 0.002,
                    ss_initial_guess = 75,
                    units = 'Newtons',
                    description = 'Thrust',
                    )
        self.alpha = Var(
                    max = float(np.radians(15)),
                    min = float(np.radians(-10)),
                    dmax = 0.03,
                    dcost = 1.4,
                    ss_initial_guess = 0.069,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.phi = Var(
                    max = float(np.radians(5)),
                    min = float(np.radians(-5)),
                    dmax = 0.0213,
                    dcost = 2,
                    ss_initial_guess = 0.034,
                    units = 'radians',
                    description = 'bank angle (roll)',
                    )
        
    # Lift coefficient data fit
    def CL(self,alpha_deg,Re):
        '''
        Returns the full wing lift coefficient given angle of attack in deg
        and Reynolds number for the reference chord
        '''
        lsa = 3.7742114873404853E-01
        lsb = 1.2431611459242210E-01
        lsc = 7.6461542484034933E-07
        lsd = -5.6822813708386566E-03
        lsf = -6.4455385473394244E-13
        lsg = -2.6505858038599027E-08
        
        return (lsa+lsb*alpha_deg+lsc*Re+lsd*alpha_deg**2+lsf*Re**2+
                lsg*alpha_deg*Re)
    
    # Drag coefficient data fit
    def CD(self,alpha_deg,Re):
        '''
        Returns the full wing drag coefficient given angle of attack in deg
        and Reynolds number for the reference chord
        '''
        dsa = 6.4481559973709565E-02
        dsb = -1.8784155229511873E-07
        dsc = 1.7932659465086720E-13
        dsd = -1.1138505718328709E-02
        dsf = 3.7504655548703261E-08
        dsg = -3.1059181794933706E-14
        dsh = 1.0975313832298184E-03
        dsi = -2.3679608054684811E-09
        dsj = 1.5846194439048924E-15
        return (dsa + dsb*Re + dsc*Re**2+dsd*alpha_deg+dsf*alpha_deg*Re+
                dsg*alpha_deg*Re**2+dsh*alpha_deg**2+dsi*alpha_deg**2*Re+
                dsj*alpha_deg**2*Re**2)
        
    # Solar panel efficiency data fit
    def panel_efficiency(self,G_sol,m=None):
        '''
        Given solar radiation in W/m^2 returns the efficiency of the solar panel at altitude
        '''
        eta = 0.12
        beta = 0.0021888986107182
        Tref = 298.15
        gamma = 0.413220518404272
        T_noct = 20.0310337470507
        G_noct = 0.519455027587048
        T_11 = 216.66
        # Check to see if we are processing Gekko variables, if so, use the gekko log10
        # otherwise use numpy
        if isinstance(G_sol,numbers.Number):
            efficiency = eta*(1-beta*(T_11-Tref+(T_noct-20)*G_sol/G_noct)+gamma*np.log10(G_sol+0.01))
        else:
            efficiency = eta*(1-beta*(T_11-Tref+(T_noct-20)*G_sol/G_noct)+gamma*m.log10(G_sol+0.01))
        return efficiency
    
    def plot_aero(self,config):
        # Lift
        x = np.degrees(np.linspace(config.aircraft.alpha.min,config.aircraft.alpha.max))
        y = np.linspace(200000,500000)
        X, Y = np.meshgrid(x, y)
        x = X.ravel()
        y = Y.ravel()
        CL_fit = np.zeros(len(x))
        CD_fit = np.zeros(len(x))
        for i in range(len(CL_fit)):
            CL_fit[i] = self.CL(x[i], y[i])
            CD_fit[i] = self.CD(x[i], y[i])
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,CL_fit)
        plt.title('CL')
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,CD_fit)
        plt.title('CD')
        
if __name__=='__main__':
    x = Aircraft_Template()