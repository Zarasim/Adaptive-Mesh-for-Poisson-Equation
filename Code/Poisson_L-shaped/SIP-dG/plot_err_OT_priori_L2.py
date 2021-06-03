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


# load error data in L2
err_0 = np.load('Data/OT/a_priori/err/L2_OT_0.0.npy')
err_1 = np.load('Data/OT/a_priori/err/L2_OT_0.4.npy')
err_2 = np.load('Data/OT/a_priori/err/L2_OT_0.667.npy')
err_3 = np.load('Data/OT/a_priori/err/L2_OT_0.8.npy')
err_4 = np.load('Data/OT/a_priori/err/L2_OT_0.933.npy')


dof_0 = np.load('Data/OT/a_priori/err/dof_OT_0.0.npy')
dof_1 = np.load('Data/OT/a_priori/err/dof_OT_0.4.npy')
dof_2 = np.load('Data/OT/a_priori/err/dof_OT_0.667.npy')
dof_3 = np.load('Data/OT/a_priori/err/dof_OT_0.8.npy')
dof_4 = np.load('Data/OT/a_priori/err/dof_OT_0.933.npy')


fig, ax = plt.subplots()
ax.plot(dof_0,err_0,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = 0.0')
ax.plot(dof_1,err_1,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = -4/5')
ax.plot(dof_2,err_2,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = -4/3')
ax.plot(dof_3,err_3,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = -8/5')
ax.plot(dof_4,err_4,linestyle = '--',marker = 'o',markersize = 3,label = ' exp = -9.3/5')


ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')


