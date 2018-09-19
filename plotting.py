import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm as colorm
import numpy as np
import pandas as pd
import glob
import os
import yaml
import seaborn as sns
import itertools

def plot3DPath(data,interpFactor=1,tight=0):
    
    x = data['x'].as_matrix() / 1000.0 # Convert to km
    y = data['y'].as_matrix() / 1000.0 # Convert to km
    z = data['h'].as_matrix() * 3.2808/1000.0 # Convert to kft
    time = data['time'].as_matrix()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    plt.plot(x, y, z, 'o')
    ax.set_zlim([40,90])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Alt (thousand feet)')
    ax.set_title('Flight Path')
    
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
    line_segs = Line3DCollection(segs,colors=colors)
    ax.set_zlim([40,90])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Alt (thousand feet)')
    ax.set_title('Flight Path')
    ax.add_collection(line_segs)
    ax.set_xlim([xs[xs>-100].min(),xs[xs<100].max()])
    ax.set_ylim([ys[ys>-100].min(),ys[ys<100].max()])
    if(tight==1):
        ax.set_zlim([zs[zs>1].min(),zs[zs<500].max()])
    else:
        ax.set_zlim([zs[zs>1].min()-5,zs[zs<500].max()+5])
    
    # Plot boundary radius
    circle = plt.Circle((0,0),3,color='orange',fill=False,linestyle='--')
    boundary = ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(boundary, z=zs.mean(), zdir="z")
    
    # Plot start and finish
    start_point = ax.scatter(xs[0,0],ys[0,0],zs[0,0],color=colors[0],marker='.')
    end_point = ax.scatter(xs[-1,1],ys[-1,1],zs[-1,1],color=colors[-1],marker='.')
    plt.legend((start_point,end_point),('Start','Finish'))
    
    return ax

def plot3DPath_NorthSouth(data,interpFactor=1,tight=0):
    
    # Switch x and y to change to north south
    x = data['y'].as_matrix() / 1000.0 # Convert to km
    y = data['x'].as_matrix() / 1000.0 # Convert to km
    z = data['h'].as_matrix() /1000.0 # Convert to km
    time = data['time'].as_matrix()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    plt.plot(x, y, z, 'o')
#    ax.set_zlim([40,90])
#    ax.set_xlabel('E (km)')
#    ax.set_ylabel('N (km)')
#    ax.set_zlabel('Alt (km)')
#    ax.set_title('Flight Path')
    
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
#    ax.set_title('Flight Path')
    ax.add_collection(line_segs)
#    ax.set_xlim([xs[xs>-100].min(),xs[xs<100].max()])
#    ax.set_ylim([ys[ys>-100].min(),ys[ys<100].max()])
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    if(tight==1):
        ax.set_zlim([zs[zs>1].min(),zs[zs<500].max()])
    else:
        ax.set_zlim([zs[zs>1].min()-5,zs[zs<500].max()+5])
#    ax.set_zlim([18,28])
    
    # Plot boundary radius
#    circle = plt.Circle((0,0),3,color='orange',fill=False,linestyle='--',linewidth=0.5)
#    boundary = ax.add_patch(circle)
#    art3d.pathpatch_2d_to_3d(boundary, z=zs.mean(), zdir="z")
    
    # Plot start and finish
    start_point = ax.scatter(xs[0,0],ys[0,0],zs[0,0],color=colors[0],marker='.')
    end_point = ax.scatter(xs[-1,1],ys[-1,1],zs[-1,1],color=colors[-1],marker='.')
    plt.legend((start_point,end_point),('Start','Finish'))
    
    ax.dist = 13
    
    return ax, fig

def plot2DPath_Labeled(data,season,config):
    
    # Switch x and y to change to north south
    x = data['y'].as_matrix() / 1000.0 # Convert to km
    y = data['x'].as_matrix() / 1000.0 # Convert to km
    z = data['h'].as_matrix() * 3.2808 # Convert to ft
    tp = data['tp'].as_matrix()
    aoa = np.degrees(data['alpha'].as_matrix())
    flux = data['g_sol'].as_matrix()
    bank = np.degrees(data['phi'].as_matrix())
    p_n = data['p_n'].as_matrix()
    drag = data['d'].as_matrix()
    p_bat = data['p_bat'].as_matrix()
    gamma = np.degrees(data['gamma'].as_matrix())
    p_solar = data['p_solar'].as_matrix()
    
    if(config['wind']['use_wind']==False):
        v = data['v'].as_matrix()
    else:
        try:
            v_a = data['v_a'].as_matrix()
            v_g = data['v_g'].as_matrix()
        except:
            v_a = np.zeros(len(x))
            v_g = np.zeros(len(x))
    
    # Calculate sun direction vector
    az = np.radians(data.azimuth.mean())
    zen = np.pi/2.0 - np.radians(data.zenith.mean())
    sun_x = np.sin(az)*np.cos(zen)
    sun_y = np.cos(az)*np.cos(zen)
    
    if(config['wind']['use_wind']==False):
        var_list = [z,v,tp,aoa,flux,bank,p_n,drag,gamma,p_bat]
        label_list = ['Height (m)','Velocity (m/s)','Thrust (N)','Angle of Attack (Deg)','Flux (W/m^2)','Bank Angle (Deg)',
                      'Power Needed (W)','Drag (N)','Gamma (Deg)','Battery Power (W)','Solar Power Recieved (W)']
    else:
        var_list = [z,v_g,v_a,tp,aoa,flux,bank,p_n,drag,gamma]
        label_list = ['Height (m)','Ground Velocity (m/s)','Air Velocity (m/s)','Thrust (N)','Angle of Attack (Deg)','Flux (W/m^2)','Bank Angle (Deg)',
                      'Power Needed (W)','Drag (N)','Gamma (Deg)']
        
    if(config['wind']['use_wind']==True):
        # Calculate wind direction vector
        v_w = np.sqrt(config['wind']['w_e']**2 + config['wind']['w_n']**2)
        w_x = config['wind']['w_e']/v_w
        w_y = config['wind']['w_n']/v_w
    
    for i, va in enumerate(var_list):
        fig = plt.figure()
        ax = plt.gca()
        
        plt.scatter(x, y, c=va, cmap=colorm.jet,vmin=va.min(),vmax=va.max(),edgecolors='none')
        cb = plt.colorbar(format='%.1f')
        cb.set_label(label_list[i])
    
        ax.set_xlabel('E (km)')
        ax.set_ylabel('N (km)')
#        ax.set_title('Flight Path')
    
        # Plot boundary radius
        circle = plt.Circle((0,0),3,color='orange',fill=False,linestyle='-')
        ax.add_patch(circle)
        
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        
        ax.quiver(2.6,-2.6,sun_x,sun_y,pivot='tail',color='k')
        ax.text(2.1,-2.4,'To Sun',fontsize=9)
        
        if(config['wind']['use_wind']==True):
            ax.quiver(-3,-3,w_x,w_y,pivot='tail',color='k')
            ax.text(-3.25,-2.7,'Wind')
            ax.text(-3.25,-3.5,str(v_w)+' m/s')
            
        direction = checkDirection(data)
        if(direction=='clockwise'):
            ax.plot([-2.6],[-2.6],marker=r'$\circlearrowright$',ms=20,c='k')
        else:
            ax.plot([-2.6],[-2.6],marker=r'$\circlearrowleft$',ms=20,c='k')
        ax.text(-2.9,-2.1,'Flight Direction',fontsize=9)
        
#        numx = len(x)/4
#        for j in range(4):
#            ax.annotate(str(j+1),(y[int(j*numx)],x[int(j*numx)]),(y[int(j*numx)],x[int(j*numx)]))
        
        plt.tight_layout()
        width  = 4.25
        height = width / 1.618
        fig.set_size_inches(width, height)
#        ax.set_title('Flight Path Hour ' +str(n/divs))
#        filename = './Results/'+folder+'/Trajectory Plots/' + '2d_path'+str(n/divs)+'_'+label_list[i].split()[0]+'.png'
        filename = '2d_path'+'_'+label_list[i].split()[0]+'_'+season+'.pdf'
        fig.savefig(filename, facecolor='none', edgecolor='none',bbox_inches='tight')
        plt.close(fig)
    
    return

def plot2DPath_NoLabel(data,season,config):
    
    # Switch x and y to change to north south
    x = data['y'].as_matrix() / 1000.0 # Convert to km
    y = data['x'].as_matrix() / 1000.0 # Convert to km
    z = data['h'].as_matrix() * 3.2808 # Convert to ft
    
    if(config['wind']['use_wind']==False):
        v = data['v'].as_matrix()
    else:
        try:
            v_a = data['v_a'].as_matrix()
            v_g = data['v_g'].as_matrix()
        except:
            v_a = np.zeros(len(x))
            v_g = np.zeros(len(x))
    
    # Calculate sun direction vector
    az = np.radians(data.azimuth.mean())
    zen = np.pi/2.0 - np.radians(data.zenith.mean())
    sun_x = np.sin(az)*np.cos(zen)
    sun_y = np.cos(az)*np.cos(zen)
    
        
    if(config['wind']['use_wind']==True):
        # Calculate wind direction vector
        v_w = np.sqrt(config['wind']['w_e']**2 + config['wind']['w_n']**2)
        w_x = config['wind']['w_e']/v_w
        w_y = config['wind']['w_n']/v_w
    
    fig = plt.figure()
    ax = plt.gca()
    
    plt.scatter(x, y,color='k')

    ax.set_xlabel('E (km)')
    ax.set_ylabel('N (km)')
#        ax.set_title('Flight Path')

    # Plot boundary radius
    circle = plt.Circle((0,0),3,color='orange',fill=False,linestyle='-')
    ax.add_patch(circle)
    
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    
    ax.quiver(2.6,-2.6,sun_x,sun_y,pivot='tail',color='k')
    ax.text(2.1,-2.4,'To Sun',fontsize=9)
    
    if(config['wind']['use_wind']==True):
        ax.quiver(-3,-3,w_x,w_y,pivot='tail',color='k')
        ax.text(-3.25,-2.7,'Wind')
        ax.text(-3.25,-3.5,str(v_w)+' m/s')
        
    direction = checkDirection(data)
    if(direction=='clockwise'):
        ax.plot([-2.6],[-2.6],marker=r'$\circlearrowright$',ms=20,c='k')
    else:
        ax.plot([-2.6],[-2.6],marker=r'$\circlearrowleft$',ms=20,c='k')
    ax.text(-2.9,-2.1,'Flight Direction',fontsize=9)
    
#        numx = len(x)/4
#        for j in range(4):
#            ax.annotate(str(j+1),(y[int(j*numx)],x[int(j*numx)]),(y[int(j*numx)],x[int(j*numx)]))
    
    plt.tight_layout()
    width  = 4.25
    height = width
    fig.set_size_inches(width, height)
#        ax.set_title('Flight Path Hour ' +str(n/divs))
#        filename = './Results/'+folder+'/Trajectory Plots/' + '2d_path'+str(n/divs)+'_'+label_list[i].split()[0]+'.png'
    filename = '2d_path'+'_'+season+'.pdf'
    fig.savefig(filename, facecolor='none', edgecolor='none',bbox_inches='tight')
    plt.close(fig)
    
    return


def plot2DPath_Radius(data,radius):
    
    x = data['x'].as_matrix()/1000
    y = data['y'].as_matrix()/1000
    aoa = np.degrees(data['alpha'].as_matrix())
    
    fig = plt.figure()
    ax = plt.gca()
    
#    plt.scatter(x, y)
    plt.scatter(x, y, c=aoa, cmap=colorm.jet,vmin=aoa.min(),vmax=aoa.max(),edgecolors='none')
    cb = plt.colorbar(format='%.1f')
    cb.set_label('Angle of Attack (deg)')
    ax.set_xlabel('E (km)')
    ax.set_ylabel('N (km)')

    # Plot boundary radius
    circle = plt.Circle((0,0),radius/1000,color='orange',fill=False,linestyle='-')
    ax.add_patch(circle)
    plt.xlim([-radius/1000,radius/1000])
    plt.ylim([-radius/1000,radius/1000])
    
    plt.tight_layout()
    width  = 4.25
    height = width / 1.618
    fig.set_size_inches(width, height)
    filename = '2d_radius'+'_'+str(radius)+'.pdf'
    fig.savefig(filename, facecolor='none', edgecolor='none',bbox_inches='tight')
    plt.close(fig)
    
    return

def plotHourly(data, numHours, plot2D, divs, folder, config, interpFactor=1, tight=0):
    
    # Make plot directory
    directory = './Results/'+folder+'/Trajectory Plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestep = data.time[1]-data.time[0]
    length = int(60*60/timestep/divs)
    
    zmin = data.h.min()
    zmax = data.h.max()
    
    for n in range(numHours*divs):
        # Plot 3D
        # Get correct length of data
        data_plot = data.iloc[n*length:(n+1)*length,:]
        # Plot 3d path and boundary
        ax, fig = plot3DPath_NorthSouth(data_plot,interpFactor,tight)
        ax.set_title('Flight Path Hour ' +str(n/divs))
        # Calculate sun direction vector
        az = np.radians(data_plot.azimuth.mean())
        zen = np.pi/2.0 - np.radians(data_plot.zenith.mean())
        sun_x = np.sin(az)*np.cos(zen)
        sun_y = np.cos(az)*np.cos(zen)
        sun_z = np.sin(zen)
        z_origin = data_plot.h.mean()* 3.2808/1000.0
#        if(tight==0):
#            ax.quiver(0,0,z_origin,sun_x,sun_y,sun_z,pivot='tail',length=3,color='k')
        filename = './Results/'+folder+'/Trajectory Plots/' + '3d_path'+str(n/divs)+'.png'
        fig.savefig(filename, facecolor='none', edgecolor='none')
#        filename = './Results/hale_2017_10_16_18_24_01 - Double alpha dmax/Plots/' + 'anim'+'_'+str(n).zfill(4)+'.png'
#        fig.savefig(filename, facecolor='none', edgecolor='none')
        plt.close(fig)
            
        # Plot 2D Version as well
        if(plot2D):
            plot2DPath_Labeled(data_plot,n,divs,folder,config)
#        ax.quiver(3,-3,sun_x,sun_y,pivot='tail',color='k')
#        ax.text(2.5,-2.7,'To Sun')
#        ax.set_title('Flight Path Hour ' +str(n))
#        fig.savefig('fft_orbits.png', facecolor='none', edgecolor='none')
#        fig.savefig('2d_path'+str(n)+'.png', facecolor='none', edgecolor='none')
        
    return

def plotHeadings(data,end):
    
    x = data['x']
    y = data['y']
    z = data['h']
    time = data['time']
    psi = data['psi']
    
    time_new = np.linspace(time[0],time[-1],len(x)*10)
    x_int = interp1d(time, x, kind='cubic')
    y_int = interp1d(time, y, kind='cubic')
    z_int = interp1d(time, z, kind='cubic')
    psi = np.array(psi)
    psi_int = interp1d(time,psi)
    u = np.cos(psi_int(time_new[0:end]))
    v = np.sin(psi_int(time_new[0:end]))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(y_int(time_new[0:end]), x_int(time_new[0:end]), z_int(time_new[0:end]), v, u, 0, length=1)
    plt.plot(y_int(time_new[0:end]),x_int(time_new[0:end]),z_int(time_new[0:end]))
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    plt.show()
    fig = plt.figure()
    plt.plot(time_new[0:end],psi_int(time_new[0:end]))
    plt.plot(time_new[0:end],x_int(time_new[0:end]))
    plt.plot(time_new[0:end],y_int(time_new[0:end]))
    plt.plot([time_new[0],time_new[end]],[3.14*2,3.14*2])
    plt.plot([time_new[0],time_new[end]],[3.14*4,3.14*4])
    plt.plot([time_new[0],time_new[end]],[3.14*6,3.14*6])
    
#def writeAnimationPath():
    #time = np.array(sol['t'])
    #time_new = np.linspace(sol['t'][0],sol['t'][-1],len(x)*10)
    #psi = np.array(sol['psi'])
    #psi_int = interp1d(time,psi)
    #theta = np.array(sol['theta'])
    #theta_int = interp1d(time,theta)
    #phi = np.array(sol['phi'])
    #phi_int = interp1d(time,phi)
    #np.savetxt('path.out',(time_new,x_int(time_new),y_int(time_new),z_int(time_new),theta_int(time_new),psi_int(time_new),phi_int(time_new)),delimiter=',')

def plotSolar(data):
    
    solar = data['p_solar']
    time = data['time']
    
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(np.array(time)/3600.0,solar)
    ax.set_title('Solar Power Received by UAV')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Solar Power Recieved (W)')
    
def plotTotalEnergy(data):
    
    te = data['te'].as_matrix() / 3.6 # Convert to kwh
    time = data['time'].as_matrix()
    
    fig = plt.figure()
    plt.plot(np.array(time)/3600.0,te)
    plt.title('Total Energy - Battery + Potential')
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Energy (kWh)')
    print("Final Total Energy: " + str(te[-1]) + " kWh")
    
def plotTotalEnergy_multiple(folderList):
    fig = plt.figure()
    # Colors
    palette = itertools.cycle(sns.color_palette())
    colors = {'140':next(palette),'130':next(palette),'120':next(palette),'110':next(palette),'100':next(palette),'90':next(palette)}
    
    for folder in folderList:
        # Load optimized data
        name = ' '.join(folder.split()[2:])
        # Extract battery mass from folder name
        bat_mass = name.split()[-2]
        color = colors[bat_mass]
        if(name.split()[0]=='SM'):
            files = glob.glob('./Results/'+folder+'/sm_results*.xlsx')
            df_sm = pd.read_excel(files[-1])
            df_sm = fix_units(df_sm)
            # Plots
            plt.plot(df_sm.time_hr,df_sm.te_kwh,':',label=name,color=color)
        else:
    #        print(name)
            files = glob.glob('./Results/'+folder+'/Intermediates/*.xlsx')
    #        print(files)
            df_opt = pd.read_excel(files[-1])
            # Load SS
            files = glob.glob('./Results/'+folder+'/ss_results*.xlsx')
            df_ss = pd.read_excel(files[-1])
            # More units
            df_opt = fix_units(df_opt)
            df_ss = fix_units(df_ss)
            # Plots
            plt.plot(df_opt.time_hr,df_opt.te_kwh,label=name+' Opt',color=color)
            plt.plot(df_ss.time_hr,df_ss.te_kwh,'--',label=name+' SS',color=color)
            print(name)
            print('Opt max: ' + str(df_opt['te_kwh'].max()))
            print('SS max: ' + str(df_ss['te_kwh'].max()))
    plt.legend()
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Energy (kWh)')
    plt.title('Total Energy vs Battery Mass')
    
def plotSOC_multiple(folderList):
    fig = plt.figure()
    df_bar = pd.DataFrame(columns=['Battery Mass','SOC','Trajectory'])
    # Colors
    palette = itertools.cycle(sns.color_palette())
    colors = {'140':next(palette),'130':next(palette),'120':next(palette),'110':next(palette),'100':next(palette),'90':next(palette)}
    
    for folder in folderList:
        # Load optimized data
        name = ' '.join(folder.split()[2:])
        # Extract battery mass from folder name
        bat_mass = name.split()[-2]
        color = colors[bat_mass]
        if(name.split()[0]=='SM'):
            files = glob.glob('./Results/'+folder+'/sm_results*.xlsx')
            df_sm = pd.read_excel(files[-1])
            df_sm = fix_units(df_sm)
            e_batt_max = df_sm['e_batt_max'].values[0]*0.277778
            df_sm['soc'] = df_sm['e_batt_kwh']/e_batt_max
            # Plots
            plt.plot(df_sm.time_hr,df_sm.soc,':',label=name,color=color)
            # Bar plot
            df_bar = df_bar.append({'Battery Mass':int(bat_mass),'SOC':df_sm['soc'].values[-1],'Trajectory':'SM'},ignore_index=True)
        else:
    #        print(name)
            files = glob.glob('./Results/'+folder+'/Intermediates/*.xlsx')
    #        print(files)
            df_opt = pd.read_excel(files[-1])
            # Load SS
            files = glob.glob('./Results/'+folder+'/ss_results*.xlsx')
            df_ss = pd.read_excel(files[-1])
            # More units
            df_opt = fix_units(df_opt)
            df_ss = fix_units(df_ss)
            e_batt_max = df_ss['e_batt_max'].values[0]*0.277778
            df_ss['soc'] = df_ss['e_batt_kwh']/e_batt_max
            df_opt['soc'] = df_opt['e_batt_kwh']/e_batt_max
            # Plots
            plt.plot(df_opt.time_hr,df_opt.soc,label=name+' Opt',color=color)
            plt.plot(df_ss.time_hr,df_ss.soc,'--',label=name+' SS',color=color)
            print(name)
            print('Opt Final: ' + str(df_opt['soc'].values[-1]))
            print('SS Final: ' + str(df_ss['te_kwh'].values[-1]))
            # Bar plot
            df_bar = df_bar.append({'Battery Mass':int(bat_mass),'SOC':df_ss['soc'].values[-1],'Trajectory':'SS'},ignore_index=True)
            df_bar = df_bar.append({'Battery Mass':int(bat_mass),'SOC':df_opt['soc'].values[-1],'Trajectory':'Opt'},ignore_index=True)
    plt.legend()
    plt.xlabel('Time (hr)')
    plt.ylabel('State of Charge')
    plt.title('State of Charge vs Battery Mass')
    # Bar chart
#    fig = plt.figure()
#    df_bar.plot.bar()
    sns.set_style("whitegrid")
    sns.factorplot(x='Battery Mass', y='SOC', hue='Trajectory', kind='bar', data=df_bar,legend=False)
    plt.legend()
    plt.xlabel('Battery Mass (kg)')
    plt.ylabel('Final State of Charge')
    
def plotAltitude_multiple(folderList):
    fig = plt.figure()
    # Colors
    palette = itertools.cycle(sns.color_palette())
    colors = {'140':next(palette),'130':next(palette),'120':next(palette),'110':next(palette),'100':next(palette),'90':next(palette)}
    
    for folder in folderList:
        # Load optimized data
        name = ' '.join(folder.split()[2:])
        # Extract battery mass from folder name
        bat_mass = name.split()[-2]
        color = colors[bat_mass]
        if(name.split()[0]=='SM'):
            files = glob.glob('./Results/'+folder+'/sm_results*.xlsx')
            df_sm = pd.read_excel(files[-1])
            df_sm = fix_units(df_sm)
            # Plots
            plt.plot(df_sm.time_hr,df_sm.h,':',label=name,color=color)
        else:
            files = glob.glob('./Results/'+folder+'/Intermediates/*.xlsx')
            df_opt = pd.read_excel(files[-1])
            # Load SS
            files = glob.glob('./Results/'+folder+'/ss_results*.xlsx')
            df_ss = pd.read_excel(files[-1])
            # More units
            df_opt = fix_units(df_opt)
            df_ss = fix_units(df_ss)
            # Plots
            plt.plot(df_opt.time_hr,df_opt.h,label=name+' Opt',color=color)
            plt.plot(df_ss.time_hr,df_ss.h,'--',label=name+' SS',color=color)
            print(name)

            
    plt.legend()
    plt.xlabel('Time (hr)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Battery Mass')
        
def fix_units(df):
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

def checkDirection(df):
    xdiff = df['x'] - df['x'].shift(1)
    ydiff = df['y'] - df['y'].shift(1)
    dot = xdiff[1:].dot(ydiff[1:])
    if(dot<0):
        direction = 'clockwise'
    else:
        direction = 'counter'
    return direction
        
    
if __name__ == '__main__':
    
#    # Select Folder
#    folder = 'hale_2017_10_25_17_25_32 - 5000 Iterations'
#    
#    # Load config
#    config_file = './Results/'+folder+'/config_file'+folder[4:].split()[0]+'.yml'
#    with open(config_file, 'r') as ifile:
#        config = yaml.load(ifile)
#        
#    # Load data
#    file_list = glob.glob('./Results/'+folder+'/Intermediates/*.xlsx')
#    data_file = file_list[-1]
#    data = pd.read_excel(data_file)
#    
#    # Plot
#    div = 1  # 1 = hourly 2 = 30 minutes
#    plot2D = False # Plot 2D state variable plots as well as 3D path
#    plotHourly(data,int(np.floor(data.time.max()/3600)),plot2D,div,folder,config)
##    plotHourly(data,2)
    
#    folderList = ['hale_2018_01_25_15_42_06 - Winter Battery 140 kg',
#                  'hale_2018_02_01_11_01_18 - SM 140 kg',
#                  'hale_2018_01_26_12_07_49 - Winter Battery 130 kg',
#                  'hale_2018_01_29_15_13_05 - SM 130 kg',
#                  'hale_2018_01_26_12_08_16 - Winter Battery 120 kg',
#                  'hale_2018_01_29_14_01_01 - SM 120 kg',
#                  'hale_2018_01_26_12_08_37 - Winter Battery 110 kg',
#                  'hale_2018_01_29_13_49_29 - SM 110 kg',
#                  'hale_2018_01_26_12_08_55 - Winter Battery 100 kg',
#                  'hale_2018_01_29_13_36_29 - SM 100 kg',
#                  'hale_2018_01_26_12_32_22 - Winter Battery 90 kg',
#                  'hale_2018_02_02_12_01_31 - SM 90 kg'
#                  ]
    folderList = ['hale_2018_04_23_07_54_18 - E216 CL 1.1 Winter',
                  'hale_2018_04_23_08_01_10 - E216 CL 1.1 Spring',
                  'hale_2018_04_23_08_01_21 - E216 CL 1.1 Summer',
                  'hale_2018_04_23_08_01_28 - E216 CL 1.1 Fall']
    plotTotalEnergy_multiple(folderList)
    plotAltitude_multiple(folderList)
    plotSOC_multiple(folderList)
