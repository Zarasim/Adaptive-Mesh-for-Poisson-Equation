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
from quality_measure import *

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


eps = 0.001
N = 10
omega = 2*pi - eps

domain_vertices = [Point(0.0, 0.0),
                   Point(1.0, 0.0),
                   Point(1.0, 1.0),
                   Point(-1.0, 1.0),
                   Point(-1.0, -1.0)]

if omega - 3.0/2.0*pi < pi/4.0:    
    domain_vertices.append((np.tan(omega - 3.0/2.0*pi), -1.0))

else:
    
    alpha = 2.0*pi - omega
    domain_vertices.append(Point(1.0, -1.0))
    domain_vertices.append(Point(1.0, -np.tan(alpha)))


geometry = mshr.Polygon(domain_vertices)


mesh_OT = mshr.generate_mesh(geometry, N) 

n_ref = 5
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
    
    # check if the point is at the boundary:
    # (near(x,0,tol) and y <= 0) or (near(y,0,tol) and x >= 0)
    #if near(x,1,tol) or near(y,1,tol) or near(x,-1,tol) or near(y,-1,tol) or 
    if (r==0):
        continue
    
    # solve OT equation 
    R = symbols('R')
    expr = R**2 + R**(1.0/2.0) - r**2
    
    t0 = time.time()
    sol = solve(expr)
    t = time.time() - t0
    tot_time +=t
    print('time for solving equation: ', t)
    R = sol[0]
    
    mesh_OT.coordinates()[i,:] = np.array([R*x/r,R*y/r])
    
    string_mesh = 'mesh_uniform/mesh_uniform_' + str(N) + '.xml.gz'
    File(string_mesh) << mesh_OT


plot(mesh_OT)
#
#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu




    
