from aircraft_template import Aircraft_Template
from utilities import Param, Var
import numpy as np

class Aircraft(Aircraft_Template):
    
    def __init__(self):
        super().__init__()
        self.name = 'aeosonde'
        
        # Battery
        self.mass_battery = Param(5,'kg')
        self.battery_energy_density = Param(350,'Whr/kg')
        self.battery_initial_SOC = Param(0.20,'N/A') # state of charge
        self.battery_max = Param(self.mass_battery.value*self.battery_energy_density.value*0.0036,'MJ')
        # Mass
        self.mass_payload = Param(5,'kg')
        self.mass_structure = Param(15,'kg')
        self.mass_total = Param(self.mass_structure.value+self.mass_battery.value
                                +self.mass_payload.value,'kg')
        # Wing
        self.chord = Param(0.19,'m')
        self.wing_top_surface_area = Param(0.55,'m^2')
        # Propeller
        self.propeller_radius = Param(0.25,'m')
        # Power
        self.motor_efficiency = Param(0.95,'N/A')
        self.power_for_payload = Param(50,'W')
        # Dynamics
        self.v = Var(
                    ss_initial_guess = 35,
                    min = 0.001,
                    units = 'm/s',
                    description = 'velocity'
                     )
        # Controls
        self.alpha = Var(
                    max = float(np.radians(30)),
                    min = float(np.radians(-30)),
                    dmax = 0.1,
                    dcost = 1.4,
                    ss_initial_guess = 0.4712,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.phi = Var(
                    max = float(np.radians(20)),
                    min = float(np.radians(-20)),
                    dmax = 0.1,
                    dcost = 1.4,
                    ss_initial_guess = 0.069,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.tp = Var(
                    max = 80,
                    min = 0.01,
                    dmax = 10,
                    dcost = 0.003,
                    ss_initial_guess = 20,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.gamma = Var( 
                    max = float(np.radians(30)),
                    min = float(np.radians(-30)),
                    up = 0.02,
                    level = 0,
                    down = -0.02,
                    mode = 'level',
                    units = 'radians',
                    description = 'flight path angle (pitch)'
                    )
        
    # Lift coefficient data fit
        
    def CL(self,alpha_deg , y_in):

        return 0.28 + 3.45 * alpha_deg * 180/np.pi
    
        
    def CD(self,alpha_deg, Re):
         CD0 = 0.03
         K = 1/(4*18.91**2*CD0)
         CL = 0.28 + 3.45 * alpha_deg * 180/np.pi
         return CD0 + K * CL**2
        
if __name__=='__main__':
    airplane = Aircraft()