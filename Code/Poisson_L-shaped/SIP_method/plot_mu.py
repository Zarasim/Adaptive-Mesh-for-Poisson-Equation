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

data_path = '/home/simone/PhD/SIP_method/Data'
pathset = os.path.join(data_path)


# load error data in H10
#
#err_unif = np.load('Data/L-shaped/unif/L2_unif.npy')
#err_0 = np.load('Data/L-shaped/h_ref/L2_href_0.0.npy')
#err_1 = np.load('Data/L-shaped/h_ref/L2_href_0.33.npy')
#err_2 = np.load('Data/L-shaped/h_ref/L2_href_0.6.npy')
##err_3 = np.load('Data/L2_href_0.8.npy')
#err_4 = np.load('Data/L-shaped/h_ref/L2_href_0.99.npy')

# upload data for r-refinement 

dof1 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.0.npy')
dof2 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.33.npy')
dof3 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.6.npy')
dof4 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.99.npy')

beta = 0.99
mu_r1 = np.load('Data/L-shaped/r-adaptive/iterations/mu_r-adaptive_' + str(beta) + '_dof_' + str(int(dof4[0])) + '.npy')
mu_r2 = np.load('Data/L-shaped/r-adaptive/iterations/mu_r-adaptive_' + str(beta) + '_dof_' + str(int(dof4[1])) + '.npy')
mu_r3 = np.load('Data/L-shaped/r-adaptive/iterations/mu_r-adaptive_' + str(beta) + '_dof_' + str(int(dof4[2])) + '.npy')
mu_r4 = np.load('Data/L-shaped/r-adaptive/iterations/mu_r-adaptive_' + str(beta) + '_dof_' + str(int(dof4[3])) + '.npy')


## For gradient and curvature


#
#dof1 = np.load('Data/L-shaped/r-adaptive/grad/dof_r-adaptive.npy')
#dof2 = np.load('Data/L-shaped/r-adaptive/grad/dof_r-adaptive.npy')
#dof3 = np.load('Data/L-shaped/r-adaptive/grad/dof_r-adaptive.npy')
#dof4 = np.load('Data/L-shaped/r-adaptive/grad/dof_r-adaptive.npy')
#
#
#mu_r1 = np.load('Data/L-shaped/r-adaptive/grad/iterations/mu_r-adaptive_dof_' + str(dof1) + '.npy')
#mu_r2 = np.load('Data/L-shaped/r-adaptive/grad/iterations/mu_r-adaptive_dof_' + str(dof2) + '.npy')
#mu_r3 = np.load('Data/L-shaped/r-adaptive/grad/iterations/mu_r-adaptive_dof_' + str(dof3) + '.npy')
#mu_r4 = np.load('Data/L-shaped/r-adaptive/grad/iterations/mu_r-adaptive_dof_' + str(dof4) + '.npy)
#
#
#mu_r1 = np.load('Data/L-shaped/r-adaptive/curv/iterations/mu_r-adaptive_dof_' + str(dof1) + '.npy')
#mu_r2 = np.load('Data/L-shaped/r-adaptive/curv/iterations/mu_r-adaptive_dof_' + str(dof2) + '.npy')
#mu_r3 = np.load('Data/L-shaped/r-adaptive/curv/iterations/mu_r-adaptive_dof_' + str(dof3) + '.npy')
#mu_r4 = np.load('Data/L-shaped/r-adaptive/curv/iterations/mu_r-adaptive_dof_' + str(dof4) + '.npy)


fig, ax = plt.subplots()
ax.plot(mu_r1[:400],linestyle = '--',marker = 'o',markersize = 5,label = 'dof: ' + str(int(dof4[0])))
ax.plot(mu_r2[:400],linestyle = '--',marker = 'o',markersize = 5,label = 'dof: ' + str(int(dof4[1])))
ax.plot(mu_r3[:400],linestyle = '--',marker = 'o',markersize = 5,label = 'dof: ' + str(int(dof4[2])))
ax.plot(mu_r4[:400],linestyle = '--',marker = 'o',markersize = 5,label = 'dof: ' + str(int(dof4[3])))

ax.set_xlabel('iterations')
ax.set_ylabel('shape regularity')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.set_title('beta = ' + str(beta))
ax.legend(loc = 'best')


