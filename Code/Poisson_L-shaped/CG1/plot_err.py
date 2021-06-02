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

data_path = '/home/simo94/PhD/Adaptive_mesh/Winslow/Poisson_L-shaped'
pathset = os.path.join(data_path)


# load error data in H10 norm
err_unif = np.load('Data/conv_test/H10_unif.npy')
err_href = np.load('Data/conv_test/H10_href.npy')
err_movmesh = np.load('Data/conv_test/H10_movmesh_a-posteriori.npy')
err_movmesh_curv = np.load('Data/conv_test/H10_movmesh_curvature.npy')
err_movmesh_grad = np.load('Data/conv_test/H10_movmesh_gradient.npy')
err_hr= np.load('Data/conv_test/H10_hr.npy')
# load dof data
dof_unif = np.load('Data/conv_test/dof_unif.npy')
dof_href = np.load('Data/conv_test/dof_href.npy')
dof_movmesh = np.load('Data/conv_test/dof_movmesh_a-posteriori.npy')
dof_movmesh_curv = np.load('Data/conv_test/dof_movmesh_curvature.npy')
dof_movmesh_grad = np.load('Data/conv_test/dof_movmesh_gradient.npy')
dof_hr = np.load('Data/conv_test/dof_hr.npy')

# load convergence rate
rate_unif = np.load('Data/conv_test/rate_unif.npy')
rate_href = np.load('Data/conv_test/rate_href.npy')
rate_movmesh = np.load('Data/conv_test/rate_movmesh_a-posteriori.npy')
rate_movmesh_curv = np.load('Data/conv_test/rate_movmesh_curvature.npy')
rate_movmesh_grad = np.load('Data/conv_test/rate_movmesh_gradient.npy')
rate_hr = np.load('Data/conv_test/rate_hr.npy')

fig, ax = plt.subplots()
ax.plot(dof_unif,err_unif,linestyle = '--',marker = 'o',markersize = 5,label = 'uniform refinement| rate %.4g' %np.mean(rate_unif))
ax.plot(dof_href,err_href,linestyle = '--',marker = 'x',markersize = 7,label = 'h-refinement| rate %.4g' %rate_href[-1])
ax.plot(dof_movmesh,err_movmesh,linestyle = '--',marker = 'v',markersize = 5,label = 'a-posteriori| rate %.4g' %np.mean(rate_movmesh))
ax.plot(dof_movmesh_curv,err_movmesh_curv,linestyle = '--',marker = '^',markersize = 5,label = 'curvature| rate %.4g' %np.mean(rate_movmesh_curv))
ax.plot(dof_movmesh_grad,err_movmesh_grad,linestyle = '--',marker = '*',markersize = 5,label = 'gradient| rate %.4g' %np.mean(rate_movmesh_grad))
ax.plot(dof_hr,err_hr,linestyle = '--',marker = '*',markersize = 5,label = 'hr| rate %.4g' %np.mean(rate_hr))


ax.set_xlabel('dof')
ax.set_ylabel('H10 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')


