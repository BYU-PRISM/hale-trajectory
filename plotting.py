# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utilities import load_pickle
import pandas as pd
import numpy as np
import glob
import os

def plot_all(folder,config=None):
    
    # If no config object is given, load from selected folder
    if config==None:
        filename = glob.glob(folder+'/*.pkl')[-1]
        config = load_pickle(filename)
    
    plt.style.use(['seaborn-paper','seaborn-whitegrid'])
    plt.rc("font", family="serif")
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    width  = 4
    height = width / 1.618
    plt.close('all')
    config.width = width
    config.height = height
    
    # Read data from folder
    data_opt = fix_units(pd.read_excel(glob.glob(folder+'/Intermediates/*.xlsx')[-1]))
    data_ss = fix_units(pd.read_excel(glob.glob(folder+'/ss*.xlsx')[-1]))
    
    # Set output folder
    plot_folder = os.path.join(config.results_folder,'Plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    config.plot_folder = plot_folder
    
    # Plot solar data
    solar_plots(data_ss,config)
    
    # Plot energy data
    energy_plots(data_opt,data_ss,config)
    
    # Plot 3D Trajectory
    trajectory_plots_3D(data_opt,config)
    
    # Plot miscelanseous plots
    misc_plots(data_opt,config)
    
def fix_units(df):
    '''
    Add additional units to the results data for plotting
    '''
    df['time_hr'] = df['time']/3600
    df['phi_deg'] = np.degrees(df['phi'])
    df['theta_deg'] = np.degrees(df['theta'])
    df['alpha_deg'] = np.degrees(df['alpha'])
    df['gamma_deg'] = np.degrees(df['gamma'])
    df['psi_deg'] = np.degrees(df['psi'])
    df['x_km'] = df['x']/1000
    df['y_km'] = df['y']/1000
    df['h_kft'] = df['h']*3.2808/1000
    df['dist_km'] = df['dist']/1000
    df['te_kwh'] = df['te']*0.277778
    df['e_batt_kwh'] = df['e_batt']*0.277778
    df['t_hr'] = df['t']/3600
    df['psi_mod'] = np.mod(df['psi'],2*np.pi)
    df['psi_deg_mod'] = np.mod(df['psi_deg'],360)
    
    # Wind
    try:
        df['gamma_a_deg'] = np.degrees(df['gamma_a'])
        df['chi_deg'] = np.degrees(df['chi'])
    except:
        pass
       
    return df

def solar_plots(data,config):
    
    # Available solar flux
    fig = plt.figure()
    df = data.copy(deep=True)
    df = df.loc[df['flux']>0.01]
    plt.plot(df.t_hr,df.flux,c='k')
    plt.ylim([0,1400])
    plt.title('Total Solar Flux Available')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Available Flux (W/m$^2$)')
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'available_flux.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
    # Solar power recieved in circular orbit
    fig = plt.figure()
    df = data.copy(deep=True)
    plt.plot(df.t_hr,df.p_solar,c='k')
    plt.title('Total Solar Flux Available')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Solar Power Recieved (W)')
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'ss_flux.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
    # Azimuth and Zenith
    df = data.copy(deep=True)
    df = df.loc[df['flux']>0.01]
    fig = plt.figure()
    plt.plot(df.time_hr,df.azimuth,c='k',label='Sun Azimuth')
    plt.plot(df.time_hr,df.zenith,'--',color="0.5",label='Sun Zenith')
    plt.legend(loc='best')
    plt.title('Solar Azimuth and Zenith')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Angle (Degrees)')
    plt.xlim([0,15])
    plt.ylim([0, 300])
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'azimuth_zenith_.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
    # Panel Efficiency
    fig = plt.figure()
    pe_list = []
    for G_sol in range(0,1500):
        panel_efficiency = config.aircraft.panel_efficiency(G_sol)
        pe_list.append(panel_efficiency)
    plt.plot(range(0,1500),pe_list,'k')
    plt.title('Solar Panel Efficiency')
    plt.xlabel('Flux (W/m^2)')
    plt.ylabel('Efficiency')
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'solar_eff.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
def energy_plots(data_opt,data_ss,config):
    
    # Total Energy
    df = data_opt.copy(deep=True)
    df_ss = data_ss.copy(deep=True)
    
    # Total Energy and Battery Energy
    fig = plt.figure()
    plt.plot(df.time_hr,df.te_kwh,label='Optimized Total Energy')
    plt.plot(df.time_hr,df.e_batt_kwh,'--',label='Optimized Battery Energy')
    plt.plot(df_ss.time_hr,df_ss.e_batt_kwh,':',label='SS Battery Energy')
    plt.legend(loc='best')
    plt.title('Energy Storage')
    plt.xlabel('Time (Hr)')
    plt.ylabel('Energy Stored (kWh)')
    plt.xlim([0,24])
    plt.ylim([0,70])
    plt.xticks([0,6,12,18,24])
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'total_energy_.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
def trajectory_plots_2D(data_opt,config):
    pass

def trajectory_plots_3D(data_opt,config):
    # Full Orbits
    ax, fig = plot3DPath_NorthSouth(data_opt,tight=True)
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'3d_path.pdf')
    plt.savefig(filepath, bbox_inches='tight')

def misc_plots(data_opt,config):
    # Elevation Profile
    fig = plt.figure()
    df = data_opt.copy(deep=True)
    plt.plot(df.time_hr,df.h/1000)
    plt.legend()
    plt.title('Optimal Trajectory Altitude')
    plt.xlabel('Time (hr)')
    plt.ylabel('Altitude (km)')
    plt.xlim([0,24])
    plt.ylim([18,25])
    plt.tight_layout()
    fig.set_size_inches(config.width, config.height)
    filepath = os.path.join(config.plot_folder,'altitude.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    
def plot3DPath_NorthSouth(data,interpFactor=1,tight=0):
    
    # Switch x and y to change to north south
    x = data['y'].as_matrix() / 1000.0 # Convert to km
    y = data['x'].as_matrix() / 1000.0 # Convert to km
    z = data['h'].as_matrix() /1000.0 # Convert to km
    time = data['time'].as_matrix()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Interpolated Path
    time_new = np.linspace(time[0],time[-1],len(x)*interpFactor)
    interp_type = 'linear'
    x_int = interp1d(time, x, kind=interp_type)
    y_int = interp1d(time, y, kind=interp_type)
    z_int = interp1d(time, z, kind='linear')
    
    MAP='winter'
    NPOINTS = len(time_new)
    cm = plt.get_cmap(MAP)
    ys = np.zeros([NPOINTS-1,2])
    xs = np.empty([NPOINTS-1,2])
    zs = np.empty([NPOINTS-1,2])
    segs = np.empty([NPOINTS-1,2,3])
    colors = []
    for i in range(NPOINTS-1):
        xs[i,:] = x_int(time_new[i:i+2])
        ys[i,:] = y_int(time_new[i:i+2])
        zs[i,:] = z_int(time_new[i:i+2])
        colors.append(cm(1.*i/(NPOINTS-1)))
    segs[:,:,0] = xs
    segs[:,:,1] = ys
    segs[:,:,2] = zs
    line_segs = Line3DCollection(segs,colors=colors,linewidth=0.5)
    ax.set_zlim([40,90])
    ax.set_xlabel('\nE (km)')
    ax.set_ylabel('\nN (km)')
    ax.set_zlabel('\nAlt (km)')
    ax.set_title('Flight Path')
    ax.add_collection(line_segs)
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    if(tight==1):
        ax.set_zlim([zs[zs>1].min(),zs[zs<500].max()])
    else:
        ax.set_zlim([zs[zs>1].min()-5,zs[zs<500].max()+5])
    
    # Plot start and finish
    start_point = ax.scatter(xs[0,0],ys[0,0],zs[0,0],color=colors[0],marker='.')
    end_point = ax.scatter(xs[-1,1],ys[-1,1],zs[-1,1],color=colors[-1],marker='.')
    plt.legend((start_point,end_point),('Start','Finish'))
    
    ax.dist = 13
    
    return ax, fig