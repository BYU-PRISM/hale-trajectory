# -*- coding: utf-8 -*-
"""
"""

from __future__ import division
from steady_state import integrate_steady_state
from settings import Settings, process_settings
from scipy.optimize import minimize
import time as tm

def final_energy_objective(m_battery,x0,config,m0,mb0):
    
    # Set battery mass to new value
    config.aircraft.mass_battery.value = m_battery
    m_new = m0 + (m_battery - mb0)
    # Set total mass to new value
    config.aircraft.mass_total.value = m_new
    
    # Scale guess values
    x0 = x0 * m_battery/mb0
    
    # Integrate circular orbit
    solData, MV, t = integrate_steady_state(config,x0)
    
    # Final battery energy
    final = solData.e_batt.iloc[-1]
    
    print('Input: ' + str(m_battery) + ' Final Energy: ' + str(final))
    
    return -final

def size_battery(config):
    # Initial guesses for thrust (N), angle of attack (rad), and bank angle (rad)
    x0 = [config.aircraft.tp.ss_initial_guess,
          config.aircraft.alpha.ss_initial_guess,
          config.aircraft.phi.ss_initial_guess]
    
    m_battery_guess = config.aircraft.mass_battery.value
    m_guess = config.aircraft.mass_total.value
    
    print('Optimizing battery size')
    sol = minimize(final_energy_objective,
                   [m_battery_guess],
                   args=(x0,config,m_guess,m_battery_guess),
                   method='Nelder-Mead',
                   options={'disp':True})
    if(sol.success==True):
        print('Succesful optimization')
    
    m_battery_opt = sol.x
    final_opt = final_energy_objective(m_battery_opt,x0,config,m_guess,m_battery_guess)
    print('M: ' + str(m_battery_opt))
    print('Final: ' + str(final_opt))
    
    return m_battery_opt, final_opt

if __name__ == '__main__':
    start = tm.time()
    
    #%% Setup
    # Load configuration settings
    config = Settings()
    
    # Process configuration settings
    config = process_settings(config)
    
    # Solve for optimum battery size
    m_battery_opt, gap_opt =size_battery(config)
    end = tm.time() - start
    print('Time: ' + str(end))