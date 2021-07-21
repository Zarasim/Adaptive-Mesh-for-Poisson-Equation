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
number_of_lines= 25
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ cm.jet(x) for x in cm_subsection ]
# read csv file and plot 

#for i,gamma in enumerate(gamma_vec[1::2]):
#    df = pd.read_csv('Data/measure_Linfty_' + str(round(gamma,2)) + '.csv')
#    measure_vec.append(df['measure'].to_numpy()[:-200]*1e16)
#    r_vec.append(df['dist'].to_numpy()[:-200])
#
# 
#fig, ax = plt.subplots()
#for i,gamma in enumerate(gamma_vec[1::2]):
#    ax.plot(r_vec[i],measure_vec[i],color = colors[i*3],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma))
#    
##for i,gamma in enumerate(gamma_vec[15:]): 
##    ax.plot(r_vec[i],ffit_vec[i],'-.',color = colors[5+i],label = ' exp = %.3g' %(coeff_vec[15+i]))
##ax.set_xlim(1e-9,3e-2)
##ax.set_ylim(1e-3,5e-1)
#ax.set_xlabel('r')
#ax.set_ylabel('measure')
#ax.set_yscale('log')
#ax.set_xscale('log')           
#ax.legend(loc = 'best')
#ax.set_title('L-infty')

###########################################################################################################################################
#

r_vec = []
measure_vec = []
ffit_vec = []
coeff_vec = []

gamma_vec = np.linspace(0.0,0.9,num)[5:]

#####  L2  ######

for i,gamma in enumerate(gamma_vec[2::2]):
    df = pd.read_csv('Data/measure_L2_' + str(round(gamma,2)) + '.csv')
    measure_vec.append(df['measure'].to_numpy()[:-200:5]*1e+6)
    r_vec.append(df['dist'].to_numpy()[:-200:5])
 #   ffit,coeff = scipy_fit(r_vec[i],measure_vec[i]*r_vec[i])
 #   ffit_vec.append(ffit)
 #   coeff_vec.append(coeff[2])


fig, ax = plt.subplots()
for i,gamma in enumerate(gamma_vec[2::2]):
    ax.plot(r_vec[i],measure_vec[i],color = colors[i],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma))
    
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


#


