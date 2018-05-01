# -*- coding: utf-8 -*-

import numpy as np

def loadWind():
    # Load wind data from file
    # This is data for Albuquerque winter solstice 2017 at 18-28 km
    uwind = np.load('uwind.npy')
    vwind = np.load('vwind.npy')
    windData = {}
    windData.uwind = uwind
    windData.vwind = vwind
    return windData

def predictWind(windData,hour):
    
    return w_n, w_e