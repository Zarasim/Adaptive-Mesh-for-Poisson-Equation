#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import mshr 
import time 
from sympy import symbols, solve
import sys

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


def quality_measure(mesh):
    
    ## Compute mesh Skewness
    mu = Function(DG0)
    
    for c in cells(mesh):       
        
        pk = c.inradius()
        hk = c.h()
        
        mu.vector()[c.index()] = pk/hk
    
    return mu


mesh_OT = Mesh('ell_mesh.xml')

mesh_OT.rotate(-90)
mesh_OT.coordinates()[:] = mesh_OT.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])


n_ref = 0
for it in range(n_ref):
    
    if it >0:
        mesh_OT = refine(mesh_OT)

    N = mesh_OT.num_vertices()


coords = mesh_OT.coordinates()[:] 

tol = 1e-12
tot_time = 0.0

for i in range(coords.shape[0]):
    
    print('iteration n°: ',i)
    # for each mesh point calculate the distance r     
    x = coords[i,0]
    y = coords[i,1]
    r = np.sqrt(x**2 + y**2)
    
    if (r==0):
        continue
    
    # solve OT equation
    R = symbols('R')
    expr = R**2 + R**(2.0/3.0) - r**2
    
    t0 = time.time()
    sol = solve(expr)
    t = time.time() - t0
    tot_time +=t
    print('time for solving equation: ', t)
    R = sol[0]
    
    mesh_OT.coordinates()[i,:] = np.array([R*x/r,R*y/r])
    
    print('total time passed: ',tot_time)
    #string_mesh = 'mesh_uniform/mesh_uniform_' + str(N) + '.xml.gz'
    #ile(string_mesh) << mesh_OT


plot(mesh_OT)
#
#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu
#
#


    
