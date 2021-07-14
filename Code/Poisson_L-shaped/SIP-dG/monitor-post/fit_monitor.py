#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:30:31 2021

@author: simone
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x,m,c,d):
    '''
    Fitting Function
    I put d as an absolute number to prevent negative values for d?
    '''
    
    return  np.power(x,m)*c + abs(d)

def scipy_fit(r,w):
    
    p0 = [-1, 1, 1]
    coeff, _ = curve_fit(func, r, w, p0) # Fit curve
    m, c, d, = coeff[0], coeff[1], coeff[2]
    
    print('m: ',m)
    print('c: ',c)
    print('d: ',abs(d))
    ffit = np.power(r,m)*c + abs(d)
    
    coeff = np.array([abs(d),c,m])
    
    return ffit,coeff
    
 
w = np.load('Data/r-adaptive/fit_data/monitor'+ str(64)  +'.npy')
r = np.load('Data/r-adaptive/fit_data/dist'+ str(64) +'.npy')


# delete first 5 elements and last 4 elements plot(r,w,marker = 'o',markersize=5,label = 'L-shaped')
# 5:29
w_min = w[1:]
r_min = r[1:]


ffit,coeff = scipy_fit(r_min,w_min)
fig, ax = plt.subplots()
ax.plot(r,w,marker = 'o',markersize=5,label = 'L-shaped')
ax.plot(r_min,w_min,marker = '*',markersize=10)
#ax.plot(r,ffit,'g-.')
ax.set_xlabel('r')
ax.set_ylabel('w')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend()
#np.save('Data/coeff2.npy',coeff)
