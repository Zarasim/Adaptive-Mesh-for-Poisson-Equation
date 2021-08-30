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
#gamma_vec = np.linspace(0.0,0.9,num)[5:-10]
gamma_vec = np.array([0.44])


r_vec = []
measure_vec = []
ffit_vec = []
coeff_vec = []

start = 0.0
stop = 1.0
number_of_lines= 30
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ cm.jet(x) for x in cm_subsection ]
# read csv file and plot 

# 35 -30
#for i,gamma in enumerate(gamma_vec[:]):
#    df = pd.read_csv('Data/measure_Linfty_' + str(round(gamma,2)) + '.csv')
#    measure_vec.append(df['measure'].to_numpy()[:-200:20])
#    r_vec.append(df['dist'].to_numpy()[:-200:20])
#
# 
#fig, ax = plt.subplots()
#for i,gamma in enumerate(gamma_vec[::2]):
#    ax.plot(r_vec[i][:80],measure_vec[i][:80],color = colors[3*i],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma_vec[i+2]))
#
##ax.set_xlim(1e-9,3e-2)
##ax.set_ylim(1e-3,5e-1)
#ax.set_xlabel('r')
#ax.set_ylabel('measure')
#ax.set_yscale('log')
#ax.set_xscale('log')           
#ax.legend(loc = 'best')
#ax.set_title('L-infty')

###########################################################################################################################################

######  L2  a-posteriori estimator ######

 
for i,gamma in enumerate(gamma_vec):
    df = pd.read_csv('Data/measure_L2_' + str(round(gamma,2)) + '.csv')
    measure_vec.append(df['measure'].to_numpy()[40::50]*1e+9)
    r_vec.append(df['dist'].to_numpy()[40::50])

fig, ax = plt.subplots()
for i,gamma in enumerate(gamma_vec):
    ax.plot(r_vec[i],measure_vec[i],color = colors[2*i],marker = 'o',markersize = 3,label = 'gamma = %.3g' %(gamma))
    
    
#ax.set_xlim(1e-9,3e-2)
#ax.set_ylim(1e-3,5e-1)
ax.set_xlabel('r')
ax.set_ylabel('measure')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')
ax.set_title('L2')
