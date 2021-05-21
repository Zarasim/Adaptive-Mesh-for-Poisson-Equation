#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:30:31 2021

@author: simone
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly

import os 

#os.getenv("HOME")
#
#data_path = '/home/simone/PhD/SIP_method/'
#pathset = os.path.join(data_path)


def personal_fit(r,w):

    ## Fit the data to a function w = A + B*r^-k
    # get A from w[0], corresponding to r = 0
    # subtract w to w[0]
    A = w[0]
    w_shift = w - A
    
    y = np.log(w[1:]/w[:-1])
    x = np.log(r[1:]/r[:-1])
    
    exponent = np.mean(y/x)
    
    ## find coefficient B by dividing the w_shift with r_shift to exponent 
    
    B = np.mean(w_shift/(r**exponent))

    ffit = A + B*r**exponent

    return ffit

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
    
    coeff = np.array([d,c,m])
    
    return ffit,coeff
    
    
def poly_fit(r,w):  
    coefs = poly.polyfit(1/r,w,5)
    ffit = poly.polyval(r,coefs)    # instead of np.poly1d
        
    return ffit


w_L = np.load('Data/r-adaptive/monitor.npy')
r_L = np.load('Data/r-adaptive/dist.npy')


# delete first 5 elements and last 4 elements 
#w = w[5:29]
#r = r[5:29]


fig, ax = plt.subplots()
ax.plot(r,w,marker = 'o',markersize=5)
ax.set_xlabel('r')
ax.set_ylabel('w')
ax.set_yscale('log')
ax.set_xscale('log')           

### call the three possible fit f
ffit_1 = personal_fit(r,w)
ffit_2,coeff = scipy_fit(r,w)
ffit_3 = poly_fit(r,w)
fig, ax = plt.subplots()
ax.plot(r,w,marker = 'o',markersize=5)
#ax.plot(r,ffit_1,'k-.')
ax.plot(r,ffit_2,'g-.')
#ax.plot(r,ffit_3,'k-.')
ax.set_xlabel('r')
ax.set_ylabel('w')
ax.set_yscale('log')
ax.set_xscale('log')           

#
#np.save('Data/coeff.npy',coeff)
#
