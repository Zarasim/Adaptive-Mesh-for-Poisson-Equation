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


# load error data in H10 norm
err_unif = np.load('Data/crack/L2_unif.npy')
err_0 = np.load('Data/crack/L2_href_0.0.npy')
#err_1 = np.load('Data/crack/L2_href_0.33.npy')
err_2 = np.load('Data/crack/L2_href_0.6.npy')
#err_3 = np.load('Data/crack/L2_href_0.8.npy')
err_4 = np.load('Data/crack/L2_href_0.99.npy')


# load dof data
dof_unif = np.load('Data/crack/dof_unif.npy')
dof_0 = np.load('Data/crack/dof_href_0.0.npy')
#dof_1 = np.load('Data/crack/dof_href_0.33.npy')
dof_2 = np.load('Data/crack/dof_href_0.6.npy')
#dof_3 = np.load('Data/crack/dof_href_0.8.npy')
dof_4 = np.load('Data/crack/dof_href_0.99.npy')

# load convergence rate
rate_unif = np.load('Data/crack/rate_unif_L2.npy')
rate_0 = np.load('Data/crack/rate_href_0.0.npy')
#rate_1 = np.load('Data/crack/rate_href_0.33.npy')
rate_2 = np.load('Data/crack/rate_href_0.6.npy')
#rate_3 = np.load('Data/crack/rate_href_0.8.npy')
rate_4 = np.load('Data/crack/rate_href_0.99.npy')


fig, ax = plt.subplots()
ax.plot(dof_unif,err_unif,linestyle = '--',marker = 'o',markersize = 5,label = 'uniform refinement| rate %.4g' %rate_unif[-1])
ax.plot(dof_0,err_0,linestyle = '--',marker = 'o',markersize = 5,label = ' beta = 0.0 | rate %.4g' %np.mean(rate_0[3:]))
#ax.plot(dof_1,err_1,linestyle = '--',marker = 'o',markersize = 5,label = ' h-ref 0.33| rate %.4g' %np.mean(rate_1[3:]))
ax.plot(dof_2,err_2,linestyle = '--',marker = 'o',markersize = 5,label = ' beta = 0.6| rate %.4g' %np.mean(rate_2[3:]))
#x.plot(dof_3,err_3,linestyle = '--',marker = 'o',markersize = 5,label = ' h-ref 0.8| rate %.4g' %np.mean(rate_3[3:]))
ax.plot(dof_4,err_4,linestyle = '--',marker = 'o',markersize = 5,label = ' beta = 0.99| rate %.4g' %np.mean(rate_4[-3:]))


ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')


