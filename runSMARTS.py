from subprocess import Popen
import subprocess
import shutil
import os.path
from os import remove, close
import numpy as np
import pandas as pd
import os
import sys
    

def pySMARTS(lat,lon,elevation,altitude,year,month,day,zone,time_range):
    
    # Initialize zenith and azimuth
    last_good_sol_zen = 0
    last_good_sol_az = 0
    
    # Loop through the day
    for hr in time_range/3600.0: 
        # Record Time
        hour = hr
    
        # Delete existing input and output files
        if os.path.isfile('pySMARTS/smarts295.inp.txt'):
            remove('pySMARTS/smarts295.inp.txt')
        if os.path.isfile('pySMARTS/smarts295.ext.txt'):
            remove('pySMARTS/smarts295.ext.txt')
        if os.path.isfile('pySMARTS/smarts295.out.txt'):
            remove('pySMARTS/smarts295.out.txt')
                
        # Copy template file
            shutil.copy('pySMARTS/inputTemplate.txt','pySMARTS/smarts295.inp.txt')
    
        # Write new input file
        replacements = {'lat':str(lat), 'lon':str(lon), 'elevation':str(elevation), 'alt':str(altitude), 'year':str(year), 'month':str(month), 'day':str(day), 'hour':str(hour), 'zone':str(zone)}
        # Python 3
        if(sys.version_info > (3, 0)):
            with open('pySMARTS/inputTemplate.txt') as infile, open('pySMARTS/smarts295.inp.txt', 'w') as outfile:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    outfile.write(line)
        # Python 2
        else:
            with open('pySMARTS/inputTemplate.txt') as infile, open('pySMARTS/smarts295.inp.txt', 'w') as outfile:
                for line in infile:
                    for src, target in replacements.iteritems():
                        line = line.replace(src, target)
                    outfile.write(line)
        
        # Run SMARTS with new input
        os.chdir('pySMARTS/')
        print('Hour: ' + str('{0:.2f}'.format(hour)))
        p = Popen("smarts295bat.exe",shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        stdout, stderr = p.communicate(input='\n\n\r'.encode())
        os.chdir('../')
        
        # Parse output file to get global irradiance for horizontal and tracking surfaces
        i = 0
        success = 0
        with open('pySMARTS/smarts295.out.txt', 'rb') as f: # This line has an error w/config file
            for line in f:
                if line.startswith(b'  Direct'):
                    lineList = line.split()
                    if(i==0):
                        horiz_global = float(lineList[9])
                        horiz_direct = float(lineList[3])
                        horiz_diffuse = float(lineList[6])
                        i = i+1
                        success = 1
                    elif(i==1):
                        track_global = float(lineList[14])
                        track_direct = float(lineList[3])
                        track_diffuse = float(lineList[7])
                        i = i+1
                if line.startswith(b'    Zenith'):
                    lineList = line.split()
                    sol_zen = float(lineList[4])
                    sol_az = float(lineList[9])
                    last_good_sol_zen = sol_zen
                    last_good_sol_az = sol_az
                if line.startswith(b'   Surface'):
                    lineList = line.split()
                    surf_tilt = float(lineList[3])
                    surf_az = float(lineList[9])
        if(success==1):
            success=0
        else:
            horiz_global = 0
            track_global = 0
            horiz_direct = 0
            horiz_diffuse = 0
            track_direct = 0
            track_diffuse = 0
            sol_zen = last_good_sol_zen # Repeat the last good zenith and azimuth to keep other things from breaking
            sol_az = last_good_sol_az
            surf_tilt = 0
            surf_az = 0
    #    print('Global Horizontal', horiz_global)
    #    print('Global Tracking', track_global)
        data_row = np.c_[hr, horiz_global, track_global, horiz_direct, track_direct, horiz_diffuse, track_diffuse, sol_zen, sol_az, surf_tilt, surf_az]
        if(hr==0):
            data = data_row
        else:
            data = np.r_[data, data_row]

    # Repair bad azimuth and zenith values from morning hours
    data[:(np.nonzero(data[:,7])[0][0]),7] = data[np.nonzero(data[:,7])[0][0],7]
    data[-100:,7] = np.linspace(data[-100,7],data[0,7],num=100)
    data[:(np.nonzero(data[:,8])[0][0]),8] = data[np.nonzero(data[:,8])[0][0],8]
    data[-100:,8] = np.linspace(data[-100,8],data[0,8],num=100)
        
    return data

if __name__ == "__main__":
    # Define input parameters
    # Albuquerque NM - Winter Solstice = Dec 12, 2016; Summer Solstice = June 20, 2016
    lat = 35.0853
    lon = -106.6056
    elevation = 1.619
    altitude = 20
    year = 2016
    month = 6 # 12
    day = 20 # 21
    zone = -7
    stepsize = 60 # 432/2
    time_range = np.arange(0,86400+stepsize,stepsize)
    solar_data = pySMARTS(lat,lon,elevation,altitude,year,month,day,zone,time_range)
    
