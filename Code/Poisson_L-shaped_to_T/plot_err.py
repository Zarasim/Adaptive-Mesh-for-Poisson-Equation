#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 3 Sept 16 09:33:45 2020

@author: simo94

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.getenv("HOME")

data_path = '/home/simone/PhD/Projects/Adaptive-Mesh-for-Poisson-Equation/Code/Poisson_L-shaped_to_T/SIP-dG'
pathset = os.path.join(data_path)

def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = np.log(err[i]/err[i+1])/(np.log(dof[i+1]/dof[i]))

    return round(rate[-1],2)

num = 7
gamma_vec = np.linspace(0.1,0.7,num)

err_vec = []
dof_vec = []

# load error data in L2
for i,gamma in enumerate(gamma_vec):
    err_vec.append(np.load('Data/Linfty_' + str(np.round(gamma_vec[i], 2)) + '.npy'))
#err_0 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[0]) + '.npy')
#err_1 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[1]) + '.npy')
#err_2 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[2]) + '.npy')
#err_3 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[3]) + '.npy')
#err_4 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[4]) + '.npy')
#err_5 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[5]) + '.npy')
#err_6 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[6]) + '.npy')
#err_7 = np.load('Data/OT/a_priori/err/L2_' + str(gamma_vec[7]) + '.npy')
    dof_vec.append(np.load('Data/dofs_' +  str(np.round(gamma_vec[i], 2)) + '.npy'))
#dof_0 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[0])  + '.npy')
#dof_1 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[1])  + '.npy')
#dof_2 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[2])  + '.npy')
#dof_3 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[3])  + '.npy')
#dof_4 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[4])  + '.npy')
#dof_5 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[5])  + '.npy')
#dof_6 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[6])  + '.npy')
#dof_7 = np.load('Data/OT/a_priori/err/dof_' + str(gamma_vec[7])  + '.npy')

fig, ax = plt.subplots()
for i in range(len(gamma_vec)):
    ax.plot(dof_vec[i],err_vec[i],linestyle = '--',marker = 'o',markersize = 3,label = ' gamma: {} | rate = {}'.format(round(gamma_vec[i],2),(conv_rate(dof_vec[i],err_vec[i]))))
#ax.plot(dof_1,err_1,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[1])
#ax.plot(dof_2,err_2,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[2])
#ax.plot(dof_3,err_3,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[3])
#ax.plot(dof_4,err_4,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[4])
#ax.plot(dof_5,err_5,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[5])
#ax.plot(dof_6,err_6,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[6])
#ax.plot(dof_7,err_7,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = %.4g' %-2*gamma_vec[7])


ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')