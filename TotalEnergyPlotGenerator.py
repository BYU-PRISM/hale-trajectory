#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:15:51 2017

@author: hunterrawson
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from shutil import copyfile

#os.chdir('/Users/hunterrawson/hale-optimization/Trajectory/Data')
file_path = os.getcwd()
os.chdir('../Simulation')
from Import_Data_New import ImportData, ImportOptimizationParameters
os.chdir('./Airfoil_Sections_Local_Flux')
from airfoil_sections import PrintTime, find_nearest
file_path = file_path + '/Data'
os.chdir(file_path)

energy_choice = 2 # 1 = MJ, 2 = kWh
def EnergyLabel(Echoice=energy_choice,Etype='Total '):
    if Echoice == 1:
        plt.ylabel(Etype + 'Energy (MJ)')
    else:
        plt.ylabel(Etype + 'Energy (kWh)')

####################################################################################################################################
data_folder_name = 'hale_2017_12_05_07_02_57 - Dragsurface5 dcost 10'
#'hale_2017_12_05_13_52_14 - Dragsurface5 dcost 10 Summer xps'
#'hale_2017_12_05_22_18_53 - Dragsurface5 dcost 10 Fall xps' 
#'' 
#'hale_2017_11_01_13_28_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg' 
#'hale_2017_11_01_16_03_24 - 15sec 10min 10shift byu ma57 Batt 139.5kg Orbit 6 km'
####################################################################################################################################

#%% Import Required Data
os.chdir('./' + data_folder_name)
data_folder_path = os.getcwd()

data_date = data_folder_name.partition(' ')[0].replace('hale_','')
config_file_name = 'config_file_' + data_date + '.yml'

if 'Fall' in data_folder_name or 'fall' in data_folder_name:
    file_season = 'Fall'
elif 'Summer' in data_folder_name or 'summer' in data_folder_name:
    file_season = 'Summer Solstice'    
elif 'Spring' in data_folder_name or 'spring' in data_folder_name:
    file_season = 'Spring'
else:
    file_season = 'Winter Solstice'
    
optimization_parameters = ImportOptimizationParameters(data_folder_path,config_file_name)
    
total_mass=optimization_parameters.total_mass
battery_mass=optimization_parameters.battery_mass
power_for_payload=optimization_parameters.power_for_payload
e_densityMJ=optimization_parameters.e_densityMJ
e_densityWhr=optimization_parameters.e_densityWhr
soc_init=optimization_parameters.initial_state_of_charge
prop_radius=optimization_parameters.propeller_radius
wing_area=optimization_parameters.wing_top_surface_area
time_step_seconds=optimization_parameters.time_step # [s]
time_step_hours=time_step_seconds/3600              # [hr]
time_array_start=optimization_parameters.start_time # [hr]
time_array_end=optimization_parameters.end_time     # [hr]
print ('\nImporting Data from ' + data_folder_name)
print ('\tTotal Mass: ' + str(total_mass) + ' ' + str(optimization_parameters.total_mass_units))
print ('\tBattery Mass: ' + str(battery_mass) + ' ' + str(optimization_parameters.battery_mass_units))
print ('\tPayload Power: ' + str(power_for_payload) + ' ' + str(optimization_parameters.payload_power_units))
print ('\tEnergy Density: ' + str(e_densityWhr) + ' Whr/kg, ' + str(e_densityMJ) + ' MJ/kg')
print ('\tInitial Battery State of Charge ' + str(soc_init))
print ('\tPropellor Radius: ' + str(prop_radius) + ' ' + str(optimization_parameters.propeller_radius_units))
print ('\tSolar Panel Surface Area: ' + str(wing_area) + ' ' + str(optimization_parameters.wing_top_surface_area_units))
print ('\tStart Time: ' + ('%.6f' % time_array_start) + ' ' + str(optimization_parameters.start_time_units))
print ('\tEnd Time: ' + ('%.6f' % time_array_end) + ' ' + str(optimization_parameters.end_time_units))
print ('\tTime Elapsed: ' + ('%.6f' % (time_array_end-time_array_start)) + ' hr')
print ('\tTime Step: ' + str(time_step_seconds) + ' ' + str(optimization_parameters.time_step_units) + ', ' + ('%.6f' % time_step_hours) + ' ' + 'hr')
soc_empty = 0.20
SOC_max = 1.0

xtix = np.array([8,12,16,20,24,28])
xtixnames = np.array(['8 AM','12 PM','4 PM','8 PM','12 AM','4AM'])
xlimit = [time_array_start,time_array_end]

opt_found = False
for i in range(0,len(os.listdir('.'))):
    if os.listdir('.')[i][0:11] == 'opt_results':
        opt_file_name = os.listdir('.')[i]
        opt_found = True
    if os.listdir('.')[i][0:10] == 'ss_results':
        ss_file_name = os.listdir('.')[i]
if opt_found == False:
    opt_file_name = 'opt_results'
    copyfile(data_folder_path + '/Intermediates/' + os.listdir(data_folder_path + '/Intermediates')[-1],data_folder_path + '/' + opt_file_name + '.xlsx')
   
opt_file_name = opt_file_name.replace('.xlsx','')
opt_file_type = 'xlsx'
ss_file_name = ss_file_name.replace('.xlsx','')
ss_file_type = 'xlsx'
ss_data_name = "Steady-state " + file_season
opt_data_name = "Optimized" + file_season

ss_elevation_header='h'
ss_elevation_unit='m'
ss_e_tot_header='te'
ss_velocity_header='v'
ss_thrust_header='thrust'
ss_bank_angle_header='phi'
ss_flight_angle_header='gamma'
ss_angle_of_attack_header='alpha'
ss_heading_header='psi'
ss_direct_tracking_flux_header='flux'
ss_time_correct_column='none'
ss_velocity_correct_column='none'
opt_time_correct_column='none'             # Used to solve issue with duplicate time columns
opt_velocity_correct_column='none'
success_header = 'none'
time_header='time'
ss_thrust_header='tp'
ss_solar_efficiency_header='panel_efficiency'
opt_elevation_header=ss_elevation_header
opt_elevation_unit=ss_elevation_unit
opt_e_tot_header=ss_e_tot_header
opt_velocity_header=ss_velocity_header
opt_thrust_header=ss_thrust_header
opt_bank_angle_header=ss_bank_angle_header
opt_flight_angle_header=ss_flight_angle_header 
opt_angle_of_attack_header=ss_angle_of_attack_header
opt_heading_header=ss_heading_header
opt_direct_tracking_flux_header=ss_direct_tracking_flux_header
ss_power_needed_header='p_n' 
ss_flux_header='g_sol' 
opt_power_needed_header = ss_power_needed_header
opt_flux_header = ss_flux_header
opt_solar_efficiency_header = ss_solar_efficiency_header
e_batt_header='e_batt'
p_batt_header='p_bat'
time_since_midnight_header='t'
success_header = 'successful_solution'
    
# Import Steady-state Data
ss_data = ImportData(FilePath=data_folder_path,FileName=ss_file_name,FileType=ss_file_type,TimeHeader=time_header,
                     ElevationHeader=ss_elevation_header,ElevationUnit=ss_elevation_unit,EBattHeader=e_batt_header,ETotHeader=ss_e_tot_header,
                     PowerNeededHeader=ss_power_needed_header,PBattHeader=p_batt_header,VelocityHeader=ss_velocity_header,
                     ThrustHeader=ss_thrust_header,FluxHeader=ss_flux_header,SolarEfficiencyHeader=ss_solar_efficiency_header,
                     BankAngleHeader=ss_bank_angle_header,FlightAngleHeader=ss_flight_angle_header,AngleOfAttackHeader=ss_angle_of_attack_header,
                     HeadingHeader=ss_heading_header,DirectTrackingFluxHeader=ss_direct_tracking_flux_header,
                     TimeCorrectColumn=ss_time_correct_column,VelocityCorrectColumn=ss_velocity_correct_column,
                     TimeSinceMidnightHeader=time_since_midnight_header)

for i in range(0,len(ss_data.time)):
    if abs(ss_data.time_since_midnight[i]-time_array_end) < 10**(-6):
        ss_ending_index = i+1
        break

ss_time = ss_data.time[:ss_ending_index]                  # (hrs)
ss_time_since_midnight = ss_data.time_since_midnight      # (hrs)
ss_flux_single = ss_data.flux[:ss_ending_index]           # (W/m^2)
ss_e_batt_data = ss_data.e_batt[:ss_ending_index]         # (MJ)
ss_p_batt = ss_data.p_batt[:ss_ending_index]              # (W)
ss_time_since_midnight = ss_data.time_since_midnight      # (hrs)
ss_elevation = ss_data.elevation[:ss_ending_index]        # (km)

# Import Optimized Data
opt_data = ImportData(FilePath=data_folder_path,FileName=opt_file_name,FileType=opt_file_type,TimeHeader=time_header,
                      ElevationHeader=opt_elevation_header,ElevationUnit=opt_elevation_unit,EBattHeader=e_batt_header,ETotHeader=opt_e_tot_header,
                      PowerNeededHeader=opt_power_needed_header,PBattHeader=p_batt_header,VelocityHeader=opt_velocity_header,
                      ThrustHeader=opt_thrust_header,FluxHeader=opt_flux_header,SolarEfficiencyHeader=opt_solar_efficiency_header,
                      BankAngleHeader=opt_bank_angle_header,FlightAngleHeader=opt_flight_angle_header,AngleOfAttackHeader=opt_angle_of_attack_header,
                      HeadingHeader=opt_heading_header,DirectTrackingFluxHeader=opt_direct_tracking_flux_header,
                      TimeCorrectColumn=opt_time_correct_column,VelocityCorrectColumn=opt_velocity_correct_column,
                      TimeSinceMidnightHeader=time_since_midnight_header,SuccessfulSolutionHeader=success_header)

# Opt Data breaks and ends early
#        for i in range(0,len(opt_data.time)):
#            if opt_data.time_since_midnight[i]*3600 == time_array_end:
#                opt_ending_index = i
#                break

opt_time = opt_data.time                    # (hrs)
opt_time_since_midnight = opt_data.time_since_midnight      # (hrs)
opt_elevation = opt_data.elevation          # (km)
opt_e_batt_data = opt_data.e_batt           # (MJ)
opt_p_batt = opt_data.p_batt                # (W)
opt_success = opt_data.success              # 1 = successful, 0 = failed
        
# Add initialization to the beginning of the opt array
if opt_time_since_midnight[0] != ss_time_since_midnight[0]:
    opt_time = np.insert(opt_time,0,ss_time[0])
    #print (opt_time_since_midnight[0],ss_time_since_midnight[0])
    opt_time_since_midnight = np.insert(opt_time_since_midnight,0,ss_time_since_midnight[0])
    #print (opt_time_since_midnight[0],ss_time_since_midnight[0])
    opt_elevation = np.insert(opt_elevation,0,ss_elevation[0])
    opt_e_batt_data = np.insert(opt_e_batt_data,0,ss_e_batt_data[0])
    opt_p_batt = np.insert(opt_p_batt,0,ss_p_batt[0])
    opt_success = np.insert(opt_success,0,1)

# Extend the opt data from the breaking point using ss data
opt_breaking_point = len(opt_time)-1
for j in range(0,len(opt_time)):
    if opt_success[j] == 0:
        #if sum(opt_success[j:j+11]) == 0: # Optimization breaks after more than 10 failures in a row
        if sum(opt_success[j:j+46]) == 0: # Optimization breaks after more than 45 failures in a row - temporary
            opt_breaking_point = j
            break
    
while len(opt_time) != len(ss_time):
    opt_time = np.append(opt_time,[opt_time[-1] + time_step_hours])
    opt_time_since_midnight = np.append(opt_time_since_midnight,[opt_time_since_midnight[-1] + time_step_hours])
 
for i in range(opt_breaking_point,len(opt_elevation)):    
    opt_elevation[i] = opt_elevation[i-1]
    opt_e_batt_data[i] = ss_e_batt_data[i]+(opt_e_batt_data[opt_breaking_point]-ss_e_batt_data[opt_breaking_point])
    opt_p_batt[i] = ss_p_batt[i]
    
for i in range(len(opt_elevation),len(opt_time)):    
    opt_elevation = np.append(opt_elevation,[opt_elevation[-5]])
    opt_e_batt_data = np.append(opt_e_batt_data,[ss_e_batt_data[i]+(opt_e_batt_data[opt_breaking_point]-ss_e_batt_data[opt_breaking_point])])
    opt_p_batt = np.append(opt_p_batt,[ss_p_batt[i]])
            
time = ss_time + time_array_start
n = len(time)

ss_sunset_point = 0
for j in range(0,len(ss_time)):
    try:
        if ss_flux_single[j] == ss_flux_single[j+1] and ss_flux_single[j] == ss_flux_single[j+15] and ss_flux_single[j] == ss_flux_single[j+35]:
            ss_sunset_point = j
            break
    except:
        ss_sunset_point = j-1
        break
        
sunset_time = time[ss_sunset_point]
print ('\tSunset Time: ' + PrintTime(sunset_time,second=True))
if time[-1]-time[0] != 24 and time[-1]-time[0] != time_array_end-time_array_start:
    print ('\n\tTime elapsed does not match time array!')

#%% Plot
def PlotTotalEnergySSVsOpt(SSE,SSSoc,OptE,OptSoc,EBattTot,MaxE,season='Winter Solstice',title_additions=''):
    plt.figure()
    EnergyLabel(Etype='Total ')
    plt.plot(time,SSE,'g',label='Steady-state')
    plt.plot(time[np.argmax(SSE)],np.amax(SSE),'go',ms=4)
    plt.plot(time,OptE,'b',label='Optimized')
    plt.plot(time[np.argmax(OptE)],np.amax(OptE),'bo',ms=4)
    plt.ylim([0,MaxE*1.07])
    [ss_minval, ss_index, ss_time_empty, opt_minval, opt_index, opt_time_empty] = [0,0,0,0,0,0] # so values can always be returned
    plt.xlim(xlimit)
    plt.xticks(xtix,xtixnames)
    [xmin, xmax, ymin, ymax] = plt.axis()
    plt.text(xmin+0.01*(xmax-xmin),MaxE+0.02*(ymax-ymin),'Max Energy: ' + str("%.2f" % MaxE))
    plt.text(xmin+0.01*(xmax-xmin),MaxE-0.1*(ymax-ymin),str("%.2f" % np.amax(SSE)),color='green',ha='left')
    plt.text(xmin+0.01*(xmax-xmin),MaxE-0.05*(ymax-ymin),str("%.2f" % np.amax(OptE)),color='blue',ha='left')
    if SSE[-1] <= soc_empty*EBattTot and OptE[-1] <= soc_empty*EBattTot:
        plt.plot(time,np.ones(n)*MaxE,'k--')
        [ss_minval, ss_index] = find_nearest(SSSoc,soc_empty,over_time_index=7/time_step_hours)
        [opt_minval, opt_index] = find_nearest(OptSoc,soc_empty,over_time_index=7/time_step_hours)
        ss_time_empty = round(time[ss_index],2)
        plt.axvline(x=ss_time_empty, ymin=-1, ymax = 1, linewidth=1, color='k')
        opt_time_empty = round(time[opt_index],2)
        plt.axvline(x=opt_time_empty, ymin=-1, ymax = 1, linewidth=1, color='k')
        plt.text(ss_time_empty-0.035*(xmax-xmin),MaxE-0.02*(ymax-ymin),'Battery Empty (SS)  ' + PrintTime(ss_time_empty),rotation=90,va='top')
        plt.text(opt_time_empty+0.01*(xmax-xmin),MaxE-0.02*(ymax-ymin),'Battery Empty (Opt)  ' + PrintTime(opt_time_empty),rotation=90,va='top')
    else:
        plt.plot(time[:int(len(time)*12/13)],np.ones(int(len(time)*12/13))*MaxE,'k--')
        if SSE[-1] > soc_empty*EBattTot and OptE[-1] <= soc_empty*EBattTot:
            plt.text(xmax-0.07*(xmax-xmin),ymax-0.02*(ymax-ymin),'Final SOC (SS):  ' + str("%.1f" % (SSSoc[-1]*100)) + ' %',rotation=90,va='top')
            [opt_minval, opt_index] = find_nearest(OptSoc,soc_empty,over_time_index=7/time_step_hours)
            opt_time_empty = round(time[opt_index],2)
            plt.axvline(x=opt_time_empty, ymin=-1, ymax = 1, linewidth=1, color='k')
            plt.text(opt_time_empty-0.035*(xmax-xmin),MaxE-0.02*(ymax-ymin),'Battery Empty (Opt)  ' + PrintTime(opt_time_empty),rotation=90,va='top')
        elif SSE[-1] <= soc_empty*EBattTot and OptE[-1] > soc_empty*EBattTot:
            plt.text(xmax-0.04*(xmax-xmin),ymax-0.02*(ymax-ymin),'Final SOC (Opt):  ' + str("%.1f" % (OptSoc[-1]*100)) + ' %',rotation=90,va='top')
            [ss_minval, ss_index] = find_nearest(SSSoc,soc_empty,over_time_index=7/time_step_hours)
            ss_time_empty = round(time[ss_index],2)
            plt.axvline(x=ss_time_empty, ymin=-1, ymax = 1, linewidth=1, color='k')
            plt.text(ss_time_empty-0.035*(xmax-xmin),MaxE-0.02*(ymax-ymin),'Battery Empty (SS)  ' + PrintTime(ss_time_empty),rotation=90,va='top')
        else:
            plt.text(xmax-0.07*(xmax-xmin),ymax-0.02*(ymax-ymin),'Final SOC (SS):  ' + str("%.1f" % (SSSoc[-1]*100)) + ' %',rotation=90,va='top')
            plt.text(xmax-0.04*(xmax-xmin),ymax-0.02*(ymax-ymin),'Final SOC (Opt):  ' + str("%.1f" % (OptSoc[-1]*100)) + ' %',rotation=90,va='top')
    plt.title(season + ' ' + title_additions)
    plt.plot(time,np.ones(n)*soc_empty*EBattTot,'k--')
    plt.axvline(x=sunset_time, ymin=-1, ymax = 1, linewidth=1, color='r')
    plt.text(sunset_time-0.035*(xmax-xmin),ymin+0.02*(ymax-ymin),'Sunset',rotation=90,fontsize=10,va='bottom')
    plt.text(sunset_time-0.035*(xmax-xmin),soc_empty*EBattTot + 0.02*(ymax-ymin),PrintTime(sunset_time),rotation=90,fontsize=10,va='bottom')
    plt.axes().yaxis.set_minor_locator(MultipleLocator(5))
    plt.axes().yaxis.set_tick_params(which='minor', right = 'on')
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)
    plt.tick_params('y',right= True, labelright=True)
    plt.legend(loc=3, ncol=1, fontsize=9.0)

    return ss_minval, ss_index, ss_time_empty, opt_minval, opt_index,opt_time_empty

g = 9.80665
h_min = 18.288   # (km) [60,000 ft]
h_max = 27.432   # (km) [90,000 ft]

e_batt_tot_energy = battery_mass*e_densityMJ # [MJ]
max_energy_raw = total_mass * g * (h_max-h_min) * 1000 / 10**6 + SOC_max * e_batt_tot_energy # [MJ]
ss_potential = (ss_elevation - h_min)*total_mass*g/1000    # [MJ]
ss_p_batt_old = ss_p_batt.copy()                     # [W]
ss_e_batt_old = ss_e_batt_data.copy()                # [MJ]
ss_e_batt_old[ss_e_batt_old>e_batt_tot_energy] = e_batt_tot_energy # correct lack of SS battery energy cap - not a completely accurate fix
ss_e_tot_old = ss_e_batt_old + ss_potential          # [MJ]
ss_soc_old = ss_e_batt_old/e_batt_tot_energy
ss_soc_old[ss_soc_old>1]=1 # correct lack of SS battery energy cap - not a completely accurate fix
opt_potential = (opt_elevation - h_min)*total_mass*g/1000  # [MJ]
opt_p_batt_old = opt_p_batt.copy()                   # [W]
opt_e_batt_old = opt_e_batt_data.copy()              # [MJ]
opt_e_tot_old = opt_e_batt_old + opt_potential       # [MJ]
opt_soc_old = opt_e_batt_old/e_batt_tot_energy

if energy_choice != 1:
    [ss_potential,ss_e_batt_old,ss_e_tot_old,opt_potential,opt_e_batt_old,
     opt_e_tot_old] = np.array([ss_potential,ss_e_batt_old,ss_e_tot_old,opt_potential,
                  opt_e_batt_old,opt_e_tot_old])/3.6 # Conversion to kWh  
    e_batt_tot_energy,max_energy_raw = np.array([e_batt_tot_energy,max_energy_raw])/3.6 # Conversion to kWh                            
                      
PlotTotalEnergySSVsOpt(ss_e_tot_old,ss_soc_old,opt_e_tot_old,opt_soc_old,e_batt_tot_energy,max_energy_raw,season=file_season,title_additions='Optimization Results')

#plt.figure()
#plt.plot(ss_e_tot_old)
#plt.plot(opt_e_tot_old)
