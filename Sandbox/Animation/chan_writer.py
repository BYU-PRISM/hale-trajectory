import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

data = pd.read_excel('hale_2017_12_22_12_16_35 - Winter single rho.xlsx')

data = data[722:809]

data = data.fillna(method='backfill')

data['y_out'] = data['y']
data['h_out'] = data['h']
data['x_out'] = data['x']

data['theta_out'] = np.degrees(data['theta'])
data['psi_out'] = np.degrees(data['psi'])
data['phi_out'] = np.degrees(data['phi'])

# Interpolate
interpFactor = 10

x = data['y'].as_matrix()
y = data['x'].as_matrix()
z = data['h'].as_matrix()
theta = data['theta_out'].as_matrix()
psi = data['psi_out'].as_matrix()
phi = data['phi_out'].as_matrix()
time = data['time'].as_matrix()

time_new = np.linspace(time[0],time[-1],len(x)*interpFactor)
interp_type = 'cubic'
x_int = interp1d(time, x, kind=interp_type)
y_int = interp1d(time, y, kind=interp_type)
z_int = interp1d(time, z, kind='linear')
psi_int = interp1d(time, psi, kind=interp_type)
phi_int = interp1d(time, phi, kind=interp_type)
theta_int = interp1d(time, theta, kind=interp_type)

xs = x_int(time_new)
ys = y_int(time_new)
zs = z_int(time_new)
thetas = theta_int(time_new)
phis = phi_int(time_new)
psis = psi_int(time_new)

dataNew = pd.DataFrame()
dataNew['y_out'] = ys
dataNew['x_out'] = xs
dataNew['h_out'] = zs
dataNew['theta_out'] = thetas
dataNew['psi_out'] = psis
dataNew['phi_out'] = phis


dataNew['index'] = np.arange(len(dataNew))+1

dataOut = dataNew[['y_out','h_out','x_out','theta_out','psi_out','phi_out']]
dataOut1 = dataNew[['index','y_out','h_out','x_out']]
dataOut2 = dataNew[['index','theta_out','psi_out','phi_out']]

#dataOutRound = dataOut.round(8)

dataOut.to_csv('path.chan',sep=' ',header=False,index=False,encoding='utf-8',line_terminator='\r\n')
dataOut1.to_csv('path_xyz.chan',sep=' ',header=False,index=False,encoding='utf-8',line_terminator='\r\n')
dataOut2.to_csv('path_phb.chan',sep=' ',header=False,index=False,encoding='utf-8',line_terminator='\r\n')

heading_x = np.cos(np.radians(dataNew['psi_out']))
heading_y = np.sin(np.radians(dataNew['psi_out']))

dataNew['cam_out_x'] = dataNew['x_out'] - heading_x * 50
dataNew['cam_out_y'] = dataNew['y_out'] - heading_y * 50
dataNew['cam_out_h'] = dataNew['h_out'] + 10
dataNew['cam_out_elevation'] = -11
dataNew['FL'] = 31.1
dataNew['roll'] = 0
dataNew['cam_az'] = dataNew['psi_out']
#data['cam_az'][3:-3] = data['cam_az'].rolling(window=7,center=True).mean()[3:-3]

dataOutCam = dataNew[['cam_out_y','cam_out_x','cam_out_h','cam_out_elevation','roll','cam_az','FL']]

dataOutCam.to_csv('cam_path.chan',sep=' ',header=False,index=False)
