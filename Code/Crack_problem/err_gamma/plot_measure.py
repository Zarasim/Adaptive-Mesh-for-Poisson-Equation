#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:43:55 2021

@author: simone
"""


import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
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
    
#    print('m: ',m)
#    print('c: ',c)
#    print('d: ',abs(d))
    ffit = np.power(r,m)*c + abs(d)
    
    coeff = np.array([abs(d),c,m])
    
    return ffit,coeff

num=30
# endpoint is not excluded
gamma_vec = np.linspace(0.0,0.9,num)[15:]
#gamma_vec[7] = 0.67


r_vec = []
measure_vec = []
ffit_vec = []
coeff_vec = []

start = 0.0
stop = 1.0
number_of_lines= 20
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ cm.jet(x) for x in cm_subsection ]
# read csv file and plot 

# 35 -30
for i,gamma in enumerate(gamma_vec[5:]):
    df = pd.read_csv('Data/measure_Linfty_' + str(round(gamma,2)) + '.csv')
    measure_vec.append(df['measure'].to_numpy()[:-20])
    r_vec.append(df['dist'].to_numpy()[:-20])
 #   ffit,coeff = scipy_fit(r_vec[i],measure_vec[i]*r_vec[i])
 #   ffit_vec.append(ffit)
 #   coeff_vec.append(coeff[2])
    

 
fig, ax = plt.subplots()
for i,gamma in enumerate(gamma_vec[5:]):
    ax.plot(r_vec[i],measure_vec[i],color = colors[i*2],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma_vec[5+i]))
    
#for i,gamma in enumerate(gamma_vec[15:]): 
#    ax.plot(r_vec[i],ffit_vec[i],'-.',color = colors[5+i],label = ' exp = %.3g' %(coeff_vec[15+i]))
#ax.set_xlim(1e-9,3e-2)
#ax.set_ylim(1e-3,5e-1)
ax.set_xlabel('r')
ax.set_ylabel('measure')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')
ax.set_title('L-infty')

###########################################################################################################################################


r_vec = []
measure_vec = []
ffit_vec = []
coeff_vec = []

gamma_vec = np.linspace(0.0,0.9,num)[13:-2]

######  L2  ######

for i,gamma in enumerate(gamma_vec[5:]):
    df = pd.read_csv('Data/measure_L2_' + str(round(gamma,2)) + '.csv')
    measure_vec.append(df['measure'].to_numpy()[:-20])
    r_vec.append(df['dist'].to_numpy()[:-20])
 #   ffit,coeff = scipy_fit(r_vec[i],measure_vec[i]*r_vec[i])
 #   ffit_vec.append(ffit)
 #   coeff_vec.append(coeff[2])


fig, ax = plt.subplots()
for i,gamma in enumerate(gamma_vec[5:]):
    ax.plot(r_vec[i],measure_vec[i],color = colors[i*2],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma_vec[i+5]))
    
#for i,gamma in enumerate(gamma_vec[15:]): 
#    ax.plot(r_vec[i],ffit_vec[i],'-.',color = colors[5+i],label = ' exp = %.3g' %(coeff_vec[15+i]))
#ax.set_xlim(1e-9,3e-2)
#ax.set_ylim(1e-3,5e-1)
ax.set_xlabel('r')
ax.set_ylabel('measure')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')
ax.set_title('L2')





