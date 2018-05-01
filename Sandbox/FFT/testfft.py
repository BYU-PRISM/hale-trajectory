import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('data.xlsx')

data_x = data['x'].as_matrix()[85:131]
data_y = data['y'].as_matrix()[85:131]
data_alpha = data['alpha'].as_matrix()[85:131]

#data_x.plot()

xfft = np.fft.rfft(data_x)
xinv = np.fft.irfft(xfft)

xd = xfft.copy()
#xd[-15:] = 0
xdinv = np.fft.irfft(xd)

yfft = np.fft.rfft(data_y)
yinv = np.fft.irfft(yfft)

yd = yfft.copy()
yd[-15:] = 0
ydinv = np.fft.irfft(yd)

plt.close('all')
plt.plot(xinv,'o')
plt.plot(xdinv)
plt.figure()
plt.plot(xfft)

plt.figure()
plt.plot(data_x,data_y)
plt.plot(xdinv,ydinv,'o')

afft = np.fft.rfft(data_alpha)
ainv = np.fft.irfft(afft)

ad = afft.copy()
#ad[-5:] = 0
adinv = np.fft.irfft(ad)

plt.figure()
plt.plot(data_alpha)
plt.plot(adinv,'o')


data_x = data['x'].as_matrix()[132:177]
data_y = data['y'].as_matrix()[132:177]
data_alpha = data['tp'].as_matrix()[132:177]

#data_x.plot()

xfft = np.fft.rfft(data_x)
xinv = np.fft.irfft(xfft)

xd2 = xfft.copy()
xd2[-15:] = 0
xdinv2 = np.fft.irfft(xd2)

yfft = np.fft.rfft(data_y)
yinv = np.fft.irfft(yfft)

yd2 = yfft.copy()
yd2[-15:] = 0
ydinv2 = np.fft.irfft(yd2)

afft = np.fft.fft(data_alpha)
ainv = np.fft.ifft(afft)

ad = afft.copy()
#ad[-5:] = 0
adinv = np.fft.ifft(ad)

plt.figure()
plt.plot(data_alpha)
plt.plot(adinv,'o')


#plt.plot(xinv,'o')
#plt.plot(xdinv)
#plt.figure()
#plt.plot(xfft)

plt.figure()
plt.plot(data_x,data_y)
plt.plot(xdinv2,ydinv2,'o')