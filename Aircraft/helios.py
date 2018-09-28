from aircraft_template import Aircraft_Template
from utilities import Param, Var
import numpy as np

class Aircraft(Aircraft_Template):
    
    def __init__(self):
        super().__init__()
        self.name = 'e216_opt'
        
        # Battery
        self.mass_battery = Param(240,'kg')
        self.battery_energy_density = Param(350,'Whr/kg')
        self.battery_initial_SOC = Param(0.20,'N/A') # state of charge
        self.battery_max = Param(self.mass_battery.value*self.battery_energy_density.value*0.0036,'MJ')
        # Mass
        self.mass_payload = Param(25,'kg')
        self.mass_structure = Param(720,'kg')
        self.mass_total = Param(self.mass_structure.value+self.mass_battery.value
                                +self.mass_payload.value,'kg')
        # Wing
        self.chord = Param(2.43,'m')
        self.wing_top_surface_area = Param(183.5,'m^2')
        # Propeller
        self.propeller_radius = Param(2,'m')
        # Power
        self.motor_efficiency = Param(0.95,'N/A')
        self.power_for_payload = Param(250,'W')
        # Dynamics
        self.v = Var(
                    ss_initial_guess = 16,
                    min = 0.001,
                    units = 'm/s',
                    description = 'velocity'
                     )
        # Controls
        self.alpha = Var(
                    max = float(np.radians(12.5)),
                    min = float(np.radians(-2)),
                    dmax = 0.03,
                    dcost = 1.4,
                    ss_initial_guess = 0.069,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.phi = Var(
                    max = float(np.radians(10)),
                    min = float(np.radians(-10)),
                    dmax = 0.03,
                    dcost = 1.4,
                    ss_initial_guess = 0.069,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        self.tp = Var(
                    max = 1000,
                    min = 0.01,
                    dmax = 25,
                    dcost = 0.003,
                    ss_initial_guess = 0.069,
                    units = 'radians',
                    description = 'angle of attack',
                    )
        
    # Lift coefficient data fit
        
    def CL(self,alpha_deg , y_in):

        return 0.0990989464607*alpha_deg + 0.0412943058344
    
        
    def CD(self,alpha_deg, Re):
         dsa = 5.8328354781017406E+03
         dsb = -1.6309313517819710E+01
         return dsa * Re**(-1) + dsb * (alpha_deg**(2) * Re**(-1))
        
if __name__=='__main__':
    airplane = Aircraft()