from subprocess import Popen
import shutil
import os.path
from os import remove, close
import numpy as np
import pandas as pd
    
# Define input parameters
# Albuquerque NM - Winter Solstice = Dec 12, 2016; Summer Solstice = June 20, 2016
lat = 35.0853
lon = -106.6056
elevation = 1.619
altitude = 25
year = 2016
month = 12
day = 21
hour = 12
zone = -7
time_interval = 5 # minutes


# Loop through the day
for hr in np.arange(0,24,time_interval/60.0):  
    # Record Time
    hour = hr

    # Delete existing input and output files
    if os.path.isfile('smarts295.inp.txt'):
        remove('smarts295.inp.txt')
    if os.path.isfile('smarts295.ext.txt'):
        remove('smarts295.ext.txt')
    if os.path.isfile('smarts295.out.txt'):
        remove('smarts295.out.txt')
            
    # Copy template file
        shutil.copy('inputTemplate.txt','smarts295.inp.txt')

    # Write new input file
    replacements = {'lat':str(lat), 'lon':str(lon), 'elevation':str(elevation), 'alt':str(altitude), 'year':str(year), 'month':str(month), 'day':str(day), 'hour':str(hour), 'zone':str(zone)}
    with open('inputTemplate.txt') as infile, open('smarts295.inp.txt', 'w') as outfile:
        for line in infile:
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
            outfile.write(line)
    
    # Run SMARTS with new input
    p = Popen("smarts295bat.exe")
    stdout, stderr = p.communicate()
    
    # Parse output file to get global irradiance for horizontal and tracking surfaces
    i = 0
    success = 0
    with open('smarts295.out.txt', 'rb') as f:
        for line in f:
            if line.startswith('  Direct'):
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
            if line.startswith('    Zenith'):
                lineList = line.split()
                sol_zen = float(lineList[4])
                sol_az = float(lineList[9])
            if line.startswith('   Surface'):
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
        sol_zen = 0
        sol_az = 0
        surf_tilt = 0
        surf_az = 0
#    print('Global Horizontal', horiz_global)
#    print('Global Tracking', track_global)
    data_row = np.c_[hr, horiz_global, track_global, horiz_direct, track_direct, horiz_diffuse, track_diffuse, sol_zen, sol_az, surf_tilt, surf_az]
    if(hr==0):
        data = data_row
    else:
        data = np.r_[data, data_row]
    
# Data to dataframe
C = ['hour', 'Global Horizonal (W/m^2)', 'Global Tracking (W/m^2)', 'Direct Horizontal', 'Direct Tracking', 'Diffuse Horizontal', 'Diffuse Tracking', 'Solar Zenith (deg)', 'Solar Azimuth (deg)', 'Surface Tilt (deg)', 'Surface Azimuth (Deg From North)']
df = pd.DataFrame(data,columns=C)

# Fix angle discontinuity
df['Solar Zenith (deg)'].iloc[-100:] = np.linspace(df['Solar Zenith (deg)'].iloc[-100],df['Solar Zenith (deg)'].iloc[0])
df['Solar Azimuth (deg)'].iloc[-100:] = np.linspace(df['Solar Azimuth (deg)'].iloc[-100],df['Solar Azimuth (deg)'].iloc[0])

# Data to file
df.to_excel("dailyOut.xls", index=False)
