# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import stats

data = pd.DataFrame()

# Get data files (from xflr5)
files = glob.glob('./e216/*')

# Read in data, get reynolds number from file name
for file in files:
    file_data = pd.read_csv(file,skiprows=6)
    name = os.path.basename(file)
    v = float(os.path.basename(file).split('-')[1].split(' ')[0].replace('_','.'))
    
#    Re = int(name[2:-4])
    rho = 0.11500
    mu = 1.422e-05
    chord = 1.41
    Re = rho*v*chord/mu
    file_data['Re'] = Re
    file_data['v'] = v
    data = data.append(file_data)
#    data = data[data['alpha']]
#    data = data[data['alpha']]
    
def dragsurface4(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 5.6446346580023496E-03
    b = -7.3998092019321339E-10
    c = -6.0276754344310316E-04
    d = 1.2513080636335438E-09
    f = 1.6614692111391766E-04
    g = -2.0587873071134208E-10

    temp += a
    temp += b * y_in
    temp += c * x_in
    temp += d * x_in * y_in
    temp += f * math.pow(x_in, 2.0)
    temp += g * math.pow(x_in, 2.0) * y_in
    return temp

def dragsurface5(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 6.3160376919356182E-03
    b = -1.5685484884476343E-09
    c = -8.8107410367649916E-04
    d = 1.3929134949446803E-09
    f = 1.9696123494676614E-04
    g = -2.0588852449107169E-10

    temp += a
    temp += b * y_in
    temp += c * x_in
    temp += d * x_in * y_in
    temp += f * math.pow(x_in, 2.0)
    temp += g * math.pow(x_in, 2.0) * y_in
    return temp

def dragsurface6(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 2.7716127593694306E-01
    b = -1.0304770820315887E+00
    c = -5.7272351899878582E-01
    Offset = 5.5119775947904334E-03

    print(x_in)
    print(b*math.pow(x_in,c))
    temp = a*math.pow(y_in,b*math.pow(x_in,c))
    temp += Offset
    return temp

def Power_PowerE_Transform_Offset_model(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 2.3270420793003342E+02
    b = -8.6660946211939216E-01
    c = -3.4019149495410661E-01
    d = -1.6361681594002206E+01
    f = 1.9444118644867066E+02
    g = 1.9546870519372906E+03
    h = -3.4918324180748940E+07
    Offset = 3.3788676147988809E-03

    temp = a * math.pow(d * x_in + f, b) * math.pow(g * y_in + h, c)
    temp += Offset
    return temp

def dragsurface7(alpha_deg,Re):
    # coefficients
    dsa = 2.3270420793003342E+02
    dsb = -8.6660946211939216E-01
    dsc = -3.4019149495410661E-01
    dsd = -1.6361681594002206E+01
    dsf = 1.9444118644867066E+02
    dsg = 1.9546870519372906E+03
    dsh = -3.4918324180748940E+07
    dsOffset = 3.3788676147988809E-03
    return dsa * (dsd * alpha_deg + dsf)**dsb * (dsg * Re + dsh)**dsc + dsOffset

def dragsurface8(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 1.3492904308341203E+01
    b = -8.0696397672456166E-01
    c = 2.2205978851294432E-02
    Offset = 5.4752618475659744E-03

    temp = a*math.pow(y_in,b+c*x_in)
    temp += Offset
    return temp

def dragsurface9(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 1.6563343560315880E+00
    b = 1.2723733405678552E+00
    c = -6.1212066019631195E-01
    Offset = 5.2683435071993884E-03

    temp = a*math.pow(b,x_in)*math.pow(y_in,c)
    temp += Offset
    return temp

def dragsurface10(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 3.3342112883090122E-02
    b = -1.9796119206322519E-03
    c = -1.0070420096893349E-07
    d = 3.7878714719930998E-04
    f = 9.7398651643931800E-14
    g = -7.7966332647571337E-11

    temp = a
    temp += b * x_in
    temp += c * y_in
    temp += d * math.pow(x_in, 2.0)
    temp += f * math.pow(y_in, 2.0)
    temp += g * x_in * y_in
    return temp

def dragsurface11(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 1.2379225459790737E+02
    b = -1.2385770684910669E+04
    c = -5.7633649503331696E-01
    d = -6.3984649510237453E-05
    f = 1.0013226926073848E+00
    g = 4.0573453980222610E+01
    h = -2.0332829937750183E+06

    temp = a * (math.pow(d * x_in + f, b) + math.pow(g * y_in + h, c))
    return temp

def dragsurfaceNACA(x_in, y_in):
    temp = 0.0

    # coefficients
    A = -1.0585858006918647E+00
    B = -1.1888373132074977E+02
    C = -1.1695866133047845E+00
    xscale = 2.3099980237773599E-04
    xoffset = 8.8273348391348150E-01
    yscale = 4.2002292769720583E+01
    yoffset = -3.6489725381467370E+06
    Offset = 8.1653544558562621E-03

    temp = math.exp(A+B*math.log(x_in * xscale + xoffset)+C*math.log(y_in * yscale + yoffset))
    temp += Offset
    return temp

def dragsurfaceE216(x_in, y_in):
    temp = 0.0

    # coefficients
    a = 9.6878064972771682E-02
    b = -1.1914394969415213E-02
    c = 1.4658946775121501E-07
    d = -7.4933620263012425E-09
    f = -1.6444247419782368E-01
    g = 3.9791780146017000E-05
    h = -4.1825694373660372E-06
    
    temp  = (a + b*x_in + c*y_in + d*x_in*y_in) / (1.0 + f*x_in + g*y_in + h*x_in*y_in)
    return temp

    

#def Polynomial_UserSelectablePolynomial_model(x_in, y_in):
#    temp = 0.0
#
#    # coefficients
#    a = 6.2590603576939555E-03
#    b = -2.0514387884365260E-09
#    c = -6.0695151626841213E-04
#    d = 1.9599583240596104E-09
#    f = 1.7449982338828179E-04
#    g = -2.5915938527308790E-10
#    h = -7.1495168670980239E-06
#    i = -1.3117148563936855E-11
#    j = 8.1822915419360012E-07
#    k = 5.4708491776387948E-13
#
#    temp += a
#    temp += b * y_in
#    temp += c * x_in
#    temp += d * x_in * y_in
#    temp += f * math.pow(x_in, 2.0)
#    temp += g * math.pow(x_in, 2.0) * y_in
#    temp += h * math.pow(x_in, 3.0)
#    temp += i * math.pow(x_in, 3.0) * y_in
#    temp += j * math.pow(x_in, 4.0)
#    temp += k * math.pow(x_in, 4.0) * y_in
#    return temp
#
#def Simple_SimpleEquation_19_Offset_model(x_in, y_in):
#    temp = 0.0
#
#    # coefficients
#    a = 2.0719944751486266E-01
#    b = 2.8449742213971110E+00
#    c = -6.0934514611797819E-01
#    Offset = 6.2070946793410181E-03
#
#    temp = a*math.pow(y_in/b,c)*math.exp(x_in/b)
#    temp += Offset
#    return temp
#
#def Power_PowerE_Transform_Offset_model(x_in, y_in):
#    temp = 0.0
#
#    # coefficients
#    a = 4.2669060804190316E-27
#    b = -8.7599207409282414E+01
#    c = 4.3776452653605119E+01
#    d = -6.7481972727065447E+00
#    f = 1.8072456724426984E+03
#    g = -5.1965911932706765E-01
#    h = 1.1046465044354532E+07
#    Offset = 6.1378808893529768E-03
#
#    temp = a * math.pow(d * x_in + f, b) * math.pow(g * y_in + h, c)
#    temp += Offset
#    return temp
#
#def pythonFit(alpha_deg,Re):
#    dsa = 6.2590603576939555E-03
#    dsb = -2.0514387884365260E-09
#    dsc = -6.0695151626841213E-04
#    dsd = 1.9599583240596104E-09
#    dsf = 1.7449982338828179E-04
#    dsg = -2.5915938527308790E-10
#    dsh = -7.1495168670980239E-06
#    dsi = -1.3117148563936855E-11
#    dsj = 8.1822915419360012E-07
#    dsk = 5.4708491776387948E-13
#    return dsa + dsb*Re + dsc*alpha_deg + dsd*alpha_deg*Re + dsf*alpha_deg**2.0 + dsg*alpha_deg**2.0*Re + dsh*alpha_deg**3.0 + dsi*alpha_deg**3.0*Re + dsj*alpha_deg**4.0 + dsk*alpha_deg**4.0*Re

def dragfit(alpha_deg,Re):
    return 3.4710563973E-07*alpha_deg**5 - 6.1283552477E-06*alpha_deg**4 + 2.6494860742E-05*alpha_deg**3 + 1.2303330810E-04*alpha_deg**2 - 5.2990511666E-04*alpha_deg + 1.0518762771E-02

#def dragsurface1(alpha_deg,Re):
#    dsa = 2.5402464438684530E-01
#    dsb = 2.7657565418069905E+00
#    dsc = -6.3815105806877470E-01
#    dsoffset = 6.4510721093745437E-03
#    return dsa * (Re/dsb)**dsc * np.exp(alpha_deg/dsb) + dsoffset
#
CDv_fit1 = np.zeros(len(data))
CDv_fit2 = np.zeros(len(data))
CDv_fit3 = np.zeros(len(data))
for i in range(len(CDv_fit1)):
    CDv_fit1[i] = dragsurfaceE216(data['alpha'].iloc[i], data['Re'].iloc[i])
#    CDv_fit2[i] = dragsurface5(data['alpha'].iloc[i], data['Re'].iloc[i])
#    CDv_fit3[i] = dragsurface4(data['alpha'].iloc[i], data['Re'].iloc[i])
    
data['CDv_fit1'] = CDv_fit1
data['CDv_fit2'] = CDv_fit2
data['CDv_fit3'] = CDv_fit3

import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_context("talk")
fig = plt.figure()
x = np.linspace(data['alpha'].min(),data['alpha'].max())
y = np.linspace(data['Re'].min(),data['Re'].max())
X, Y = np.meshgrid(x, y)
Z = dragsurfaceE216(X.ravel(),Y.ravel())
Z = Z.reshape(X.shape)
plt.pcolormesh(X,Y,Z)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Cd_v', rotation=270)
plt.xlabel('Alpha (deg)')
plt.ylabel('Re')
#plt.title('Parasitic Drag Polar')
plt.legend()
plt.tight_layout()
width  = 5
height = width / 1.618
fig.set_size_inches(width, height)
plt.savefig('drag_coeff.pdf', bbox_inches='tight')

#Re_more = np.linspace(data['Re'].iloc[0],data['Re'].iloc[-1])
#CDv_fit3_more = []
#alpha_more = np.linspace(-1,10)
#re_more_list = []
#alpha_more_list = []
#for i in range(len(Re_more)):
#    for j in range(len(alpha_more)):
#        re_more_list.append(Re_more[i])
#        alpha_more_list.append(alpha_more[j])
#        CDv_fit3_more.append(dragsurface4(alpha_more[j], Re_more[i]))
#    
#data['CDv_fit3_more'] = CDv_fit3_more
#
# Plot data
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(data['alpha'],data['Re'],data[' CL'])
#ax.set_xlabel('Alpha')
#ax.set_ylabel('Re')
#ax.set_zlabel('CL')
#plt.title('CL')

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data['alpha'],data['Re'],data[' CDv'],label='Data')
#ax.scatter(data['alpha'],data['v'],data[' CDv'],label='Data')
ax.scatter(data['alpha'],data['Re'],data['CDv_fit1'],label='Fit')
#ax.scatter(data['alpha'],data['Re'],data['CDv_fit2'],label='5')
#ax.scatter(data['alpha'],data['Re'],data['CDv_fit3'],label='4')
#ax.scatter(alpha_more_list,re_more_list,CDv_fit3_more,label='Poly Fill')
ax.set_xlabel('Alpha')
ax.set_ylabel('Re')
ax.set_zlabel('CDv')
plt.title('CDv')
plt.legend()

# 2D Plots
#fig = plt.figure()
#Res = data['Re'].unique()
#for Re in Res:
#    data_filtered = data[data['Re']==Re]
#    plt.plot(data_filtered['alpha'],data_filtered['CDv_fit1'],label='Re='+str(Re))
#plt.xlabel('Alpha')
#plt.ylabel('CDv')
#plt.title('Viscous Drag Polar Fit')
#plt.legend()
#
# 2D Plots
fig = plt.figure()
Res = data['Re'].unique()
#Res = np.linspace(80000,600000,10)
for Re in Res:
    data_filtered = data[data['Re']==Re]
    plt.plot(data_filtered['alpha'],data_filtered['CDv_fit1'],label='Re='+str(Re))
#for Re in Res:
#    data_filtered = data[data['Re']==Re]
#    plt.plot(data_filtered['alpha'],data_filtered['CDv_fit1'],label='Re='+str(Re),color='red')
#data_filtered = data[data['Re']==276788]
#plt.scatter(data_filtered['alpha'],data_filtered[' CDv'],label='Re='+str(Re),color='blue',linewidth=4)
#plt.plot(data_filtered['alpha'],data_filtered['CDv_fit'],label='Re='+str(Re),color='red',linewidth=4)
plt.xlabel('Alpha')
plt.ylabel('CDv')
plt.title('Parasitic Drag Polar')
plt.legend()
plt.tight_layout()
width  = 5
height = width / 1.618
fig.set_size_inches(width, height)
#plt.savefig('power_inout.pdf', bbox_inches='tight')

fig = plt.figure()
Res = data['Re'].unique()
#Res = np.linspace(80000,600000,10)
for Re in Res:
    data_filtered = data[data['Re']==Re]
    plt.plot(data_filtered['alpha'],data_filtered[' CL'],label='Re='+str(Re))
plt.xlabel('Alpha (deg)')
plt.ylabel('CL')
plt.xlim(data['alpha'].min(),data['alpha'].max())
plt.ylim(data_filtered[' CL'].min(),data_filtered[' CL'].max())
#plt.title('Wing Lift Coefficient')
plt.tight_layout()
width  = 5
height = width / 1.618
fig.set_size_inches(width, height)
plt.savefig('wing_lift.pdf', bbox_inches='tight')
#
# Linear fit for cl
data_filtered = data[data['Re']==data['Re'].iloc[0]]
slope, intercept, r_value, p_value, std_err = stats.linregress(data_filtered['alpha'],data_filtered[' CL'])
#    
#
CDout = data[['alpha','Re',' CDv']]
#
CDout.to_csv('cdv.csv',header=False,index=False)