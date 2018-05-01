# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from findOrbits import find_orbits
import matplotlib.pyplot as plt

# Load trajectory data
data = pd.read_excel('dataLongNoCircle.xlsx')

# Split into orbits
x_orbits, y_orbits, z_orbits, t_orbits = find_orbits(data)

# Let's get rid of the first two - hasn't formed a consistent shape yet.
x_orbits = x_orbits[2:]
y_orbits = y_orbits[2:]
z_orbits = z_orbits[2:]
t_orbits = t_orbits[2:]

# Take fft of each orbit individually - x
x_coeffs = []
x_paths = []
for x in x_orbits:
    xfft = np.fft.rfft(x) # Fourier transform
#    xfft[-15:] = 0 # Set last 15 coefficients to zero
    xdinv = np.fft.irfft(xfft) # Path back to time domain
    x_coeffs.append(xfft) # Save coefficients
    x_paths.append(xdinv) # Save new version of path

# Take fft of each orbit individually - y
y_coeffs = []
y_paths = []
for y in y_orbits:    
    yfft = np.fft.rfft(y) # Fourier transform
#    yfft[-15:] = 0 # Set last 15 coefficients to zero
    ydinv = np.fft.irfft(yfft) # Path back to time domain
    y_coeffs.append(yfft) # Save coefficients
    y_paths.append(ydinv) # Save new version of path
    
# Take fft of each orbit individually - z
z_coeffs = []
z_paths = []
for z in z_orbits:    
    zfft = np.fft.fft(z) # Fourier transform
#    zfft[-15:] = 0 # Set last 15 coefficients to zero
    zdinv = np.fft.ifft(zfft) # Path back to time domain
    z_coeffs.append(zfft) # Save coefficients
    z_paths.append(zdinv) # Save new version of path

# Plot Original Orbits and Reduced FFT Fits
plt.close('all')
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
fig.suptitle('Orbits with FFT Fits', fontsize=16)

# X and Y
for i in range(15): # First 15 orbits (~2 hours)
    axs[i].plot(x_orbits[i],y_orbits[i],'.')
    axs[i].plot(x_paths[i],y_paths[i])
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
    axs[i].set_title("{:.1f}".format(t_orbits[i]/3600)+' hrs')
    
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig('orbits.png', facecolor='none', edgecolor='none')
   

# Plot Z for orbits
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
plt.suptitle('Z with FFT Fit')

for i in range(15):
    axs[i].plot(range(len(z_orbits[i])),z_orbits[i],'.')
    axs[i].plot(range(len(z_paths[i])),z_paths[i])
    
# Plot Fourier coefficients - x
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
plt.suptitle('X Fourier coefficients')

for i in range(15):
    axs[i].plot(x_coeffs[i])
    
# Plot Fourier coefficients - y
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
plt.suptitle('Y Fourier coefficients')

for i in range(15):
    axs[i].plot(y_coeffs[i])
    
# Plot Fourier coefficients - z
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
plt.suptitle('Z Fourier coefficients')

for i in range(15):
    axs[i].plot(z_coeffs[i])
        
# Collect FFT coefficients by time instead of by orbit
# i.e. the first coefficient from each orbit, and then the second ect.
# Curve fit and plot 
fig, axs = plt.subplots(5,2)
axs = axs.ravel()
fig.suptitle('The first 10 X FFT coefficients over time', fontsize=16)

x_s_list = []
px_list = []
for i in range(10): # First ten fft coefficients
    x_s = []
    for j in range(15): # First ten orbits
        x_s.append(x_coeffs[j][i]) # Collect the i'th coefficient from all orbits
    x_s_list.append(x_s)
    axs[i].plot(x_s)
    px = np.poly1d(np.polyfit(range(15),x_s_list[i],3)) # Fit i'th coeff with cubic polynomial
    px_list.append(px)
    axs[i].plot(px(range(15)))
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
    axs[i].set_title(str(i))
    
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig('x_coeff_fit.png', facecolor='none', edgecolor='none')
    
# Repeat for y
fig, axs = plt.subplots(5,2)
axs = axs.ravel()
plt.suptitle('The first 10 Y FFT coefficients over time')
    
y_s_list = []
py_list = []
for i in range(10):
    y_s = []
    for j in range(15):
        y_s.append(y_coeffs[j][i])
    y_s_list.append(y_s)
    axs[i].plot(y_s)
    py = np.poly1d(np.polyfit(range(15),y_s_list[i],3))
    py_list.append(py)
    axs[i].plot(py(range(15)))    

# Get average length of fft arrays (This could be improved, maybe curve fit the lengths as well)
coeff_length = 0
length_list = []
for coeffs in x_coeffs:
    coeff_length = coeff_length + len(coeffs)
    length_list.append(len(coeffs))
avg_length = int(coeff_length/len(x_coeffs))
length_poly = np.poly1d(np.polyfit(range(len(length_list)),length_list,2))

# Curve fit to fft array lengths
plt.figure()
plt.plot(range(15),length_poly(range(15)))
plt.plot(range(15),length_list)
plt.title('Curve fit to array lengths')

# Plot interpolated fits for first 15 orbits
fig, axs = plt.subplots(3,5)
axs = axs.ravel()
fig.suptitle('Interpolated Fourier Orbits', fontsize=16)

for tm in range(15):
    
    # Initialize arrays to hold interpolated fourier coefficients
    new_x_fft = np.zeros(int(length_poly(tm)),dtype=np.complex)
    new_y_fft = np.zeros(int(length_poly(tm)),dtype=np.complex)

    # Get interpolated coefficients for that time
    for i in range(10):
        new_x_fft[i] = px_list[i](tm)
        new_y_fft[i] = py_list[i](tm)
    
    # Transform interpolated coefficients back to time domain    
    new_x = np.fft.irfft(new_x_fft)
    new_y = np.fft.irfft(new_y_fft)
    
    # Plot original vs interpoalted orbits
    l1 = axs[tm].plot(x_orbits[tm],y_orbits[tm],'.',label='MPC')
    l2 = axs[tm].plot(new_x,new_y,label='FFT Predicted')
    axs[tm].set_xticklabels([])
    axs[tm].set_yticklabels([])
    axs[tm].set_title("{:.1f}".format(t_orbits[tm]/3600)+' hrs')
    
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig('fft_orbits.png', facecolor='none', edgecolor='none')