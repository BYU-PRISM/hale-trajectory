# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev

data = pd.read_excel('data.xlsx')

data_x = data['x'].as_matrix()[85:131]
data_y = data['y'].as_matrix()[85:131]
data_t = data['time'].as_matrix()[85:131]


s_x = splrep(data_t,data_x)
s_y = splrep(data_t,data_y)

x2 = splev(data_t,s_x)
y2 = splev(data_t,s_y)

plt.figure()
plt.scatter(data_t,data_x)
plt.plot(data_t,x2)

plt.figure()
plt.scatter(data_t,data_y)
plt.plot(data_t,y2)

plt.figure()
plt.scatter(data_x,data_y)
plt.plot(x2,y2)

data_alpha = data['alpha'].as_matrix()[85:131]

s_alpha = splrep(data_t,data_alpha)
alpha_2 = splev(data_t,s_alpha)

plt.figure()
plt.scatter(data_t,data_alpha)
plt.plot(data_t,alpha_2)