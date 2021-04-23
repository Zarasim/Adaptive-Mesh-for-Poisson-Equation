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

err_unif = np.load('Data/L-shaped/unif/L2_unif.npy')
err_0 = np.load('Data/L-shaped/h_ref/GERS/L2_href_0.0.npy')
err_1 = np.load('Data/L-shaped/h_ref/GERS/L2_href_0.33.npy')
err_2 = np.load('Data/L-shaped/h_ref/GERS/L2_href_0.5.npy')
#err_3 = np.load('Data/L2_href_0.8.npy')
err_4 = np.load('Data/L-shaped/h_ref/GERS/L2_href_0.99.npy')

err_0M = np.load('Data/L-shaped/h_ref/MS/L2_href_0.0.npy')
err_1M = np.load('Data/L-shaped/h_ref/MS/L2_href_0.33.npy')
err_2M = np.load('Data/L-shaped/h_ref/MS/L2_href_0.5.npy')
#err_3 = np.load('Data/L2_href_0.8.npy')
err_4M = np.load('Data/L-shaped/h_ref/MS/L2_href_0.99.npy')


# upload data for r-refinement 
err_r1 = np.load('Data/L-shaped/r-adaptive/L2_r-adaptive_0.0.npy')
err_r2 = np.load('Data/L-shaped/r-adaptive/L2_r-adaptive_0.33.npy')
err_r3 = np.load('Data/L-shaped/r-adaptive/L2_r-adaptive_0.6.npy')
err_r4 = np.load('Data/L-shaped/r-adaptive/L2_r-adaptive_0.99.npy')


#err_grad = np.load('Data/L-shaped/r-adaptive/grad/L2_r-adaptive.npy')
#err_curv = np.load('Data/L-shaped/r-adaptive/curv/L2_r-adaptive.npy')


# load dof data
dof_unif = np.load('Data/L-shaped/unif/dof_unif.npy')
dof_0 = np.load('Data/L-shaped/h_ref/GERS/dof_href_0.0.npy')
dof_1 = np.load('Data/L-shaped/h_ref/GERS/dof_href_0.33.npy')
dof_2 = np.load('Data/L-shaped/h_ref/GERS/dof_href_0.5.npy')
#dof_3 = np.load('Data/dof_href_0.8.npy')
dof_4 = np.load('Data/L-shaped/h_ref/GERS/dof_href_0.99.npy')

dof_0M = np.load('Data/L-shaped/h_ref/MS/dof_href_0.0.npy')
dof_1M = np.load('Data/L-shaped/h_ref/MS/dof_href_0.33.npy')
dof_2M = np.load('Data/L-shaped/h_ref/MS/dof_href_0.5.npy')
#dof_3 = np.load('Data/dof_href_0.8.npy')
dof_4M = np.load('Data/L-shaped/h_ref/MS/dof_href_0.99.npy')

# upload data from r-refinement 
dof_r1 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.0.npy')
dof_r2 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.33.npy')
dof_r3 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.6.npy')
dof_r4 = np.load('Data/L-shaped/r-adaptive/dof_r-adaptive_0.99.npy')

#dof_grad = np.load('Data/L-shaped/r-adaptive/grad/dof_r-adaptive.npy')
#dof_curv = np.load('Data/L-shaped/r-adaptive/curv/dof_r-adaptive.npy')


# load convergence rate
rate_unif = np.load('Data/L-shaped/unif/rate_unif_L2.npy')
rate_0 = np.load('Data/L-shaped/h_ref/GERS/rate_href_0.0.npy')
rate_1 = np.load('Data/L-shaped/h_ref/GERS/rate_href_0.33.npy')
rate_2 = np.load('Data/L-shaped/h_ref/GERS/rate_href_0.5.npy')
#rate_3 = np.load('Data/rate_href_0.8.npy')
rate_4 = np.load('Data/L-shaped/h_ref/GERS/rate_href_0.99.npy')


# upload data from r-refinement 
rate_r1 = np.load('Data/L-shaped/r-adaptive/rate_r-adaptive_0.0.npy')
rate_r2 = np.load('Data/L-shaped/r-adaptive/rate_r-adaptive_0.33.npy')
rate_r3 = np.load('Data/L-shaped/r-adaptive/rate_r-adaptive_0.6.npy')
rate_r4 = np.load('Data/L-shaped/r-adaptive/rate_r-adaptive_0.99.npy')

#rate_grad = np.load('Data/L-shaped/r-adaptive/grad/rate_r-adaptive.npy')
#rate_curv = np.load('Data/L-shaped/r-adaptive/curv/rate_r-adaptive.npy')


fig, ax = plt.subplots()
ax.plot(dof_unif,err_unif,linestyle = '--',marker = 'o',markersize = 3,label = 'uniform refinement| rate %.4g' %rate_unif[-1])
ax.plot(dof_0,err_0,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.0 | rate %.4g' %np.mean(rate_0[-10:]))
ax.plot(dof_1,err_1,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.33 | rate %.4g' %np.mean(rate_1[-10:]))
ax.plot(dof_2,err_2,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.5 | rate %.4g' %np.mean(rate_2[-10:]))
##ax.plot(dof_3,err_3,linestyle = '--',marker = 'o',markersize = 5,label = ' h-ref 0.8| rate %.4g' %np.mean(rate_3[3:]))
ax.plot(dof_4,err_4,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.99 | rate %.4g' %np.mean(rate_4[-10:]))


ax.plot(dof_0M,err_0M,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.0M | rate %.4g' %np.mean(rate_0[-10:]))
ax.plot(dof_1M,err_1M,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.33M | rate %.4g' %np.mean(rate_1[-10:]))
ax.plot(dof_2M,err_2M,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.5M | rate %.4g' %np.mean(rate_2[-10:]))
##ax.plot(dof_3,err_3,linestyle = '--',marker = 'o',markersize = 5,label = ' h-ref 0.8| rate %.4g' %np.mean(rate_3[3:]))
ax.plot(dof_4M,err_4M,linestyle = '--',marker = 'o',markersize = 3,label = ' beta = 0.99M | rate %.4g' %np.mean(rate_4[-10:]))

#ax.plot(dof_r1,err_r1[:-1],linestyle = '-.',marker = '^',markersize = 3,label = 'r-adaptive | beta = 0.0')
#ax.plot(dof_r2,err_r2,linestyle = '-.',marker = '^',markersize = 3,label = 'r-adaptive |beta = 0.33')
#ax.plot(dof_r3,err_r3,linestyle = '-.',marker = '^',markersize = 3,label = 'r-adaptive |beta = 0.6')
#ax.plot(dof_r4,err_r4,linestyle = '-.',marker = '^',markersize = 3,label = 'r-adaptive |beta = 0.99')


#
#ax.plot(dof_grad,err_grad,linestyle = '--',marker = '^',markersize = 5,label = 'r-adaptive grad')
#ax.plot(dof_curv,err_curv,linestyle = '--',marker = '^',markersize = 5,label = 'r-adaptive curv')


ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')


