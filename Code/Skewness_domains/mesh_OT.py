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
from sympy import symbols
import sympy
import math
from quality_measure import *

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


eps = 0.001
#omega = 2*pi - eps

omega = 2*pi - eps
gamma = 1 - pi/(2*omega)

N = 10
domain_vertices = [Point(0.0, 0.0),
                   Point(1.0, 0.0),
                   Point(1.0, 1.0),
                   Point(-1.0, 1.0),
                   Point(-1.0, -1.0)]

if omega - 3.0/2.0*pi < pi/4.0:    
    domain_vertices.append(Point(np.tan(omega - 3.0/2.0*pi), -1.0))

else:
    
    alpha = 2.0*pi - omega
    domain_vertices.append(Point(1.0, -1.0))
    domain_vertices.append(Point(1.0, -np.tan(alpha)))


geometry = mshr.Polygon(domain_vertices)


mesh_c = mshr.generate_mesh(geometry, N) 
mesh_OT = mshr.generate_mesh(geometry, N) 

n_ref = 0
for it in range(n_ref):
    
    if it >0:
        mesh_OT = refine(mesh_OT)

    N = mesh_OT.num_vertices()

coords = mesh_OT.coordinates()[:] 

tol = 1e-12
tot_time = 0.0



for i in range(coords.shape[0]):
    
    # for each mesh point calculate the distance r     
    x = coords[i,:]
    #y = coords[i,1]
    s = np.sqrt(x[0]**2 + x[1]**2)
    
    # check if the point is at the boundary:
    # (near(x,0,tol) and y <= 0) or (near(y,0,tol) and x >= 0)
    #if near(x,1,tol) or near(y,1,tol) or near(x,-1,tol) or near(y,-1,tol) or 
    if (s==0):
        continue

    theta = math.atan2(abs(x[1]),abs(x[0]))
    
    if x[0] < 0 and x[1] > 0:
        theta = pi - theta
        
    elif x[0] <= 0 and x[1] <= 0:
        theta = pi + theta
    
    elif x[0] >= 0 and x[1] <= 0:
        theta = 2*pi - theta
        
    # x[0] = 1
    if (theta >= 0) and (theta <= pi/4):
        length_side = sqrt(1 + math.tan(theta)**2)
    # x[1] = 1
    elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
        length_side = sqrt(1 + (1/math.tan(theta))**2)
    # x[0] = -1
    elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
        length_side = sqrt(1 + math.tan(theta)**2)
    # x[1] = -1 
    elif (theta > 5.0/4.0*pi) and (theta <= 7.0/4.0*pi):
        length_side = sqrt(1 + (1/math.tan(theta))**2)
    
    elif theta > 7.0/4.0*pi:
        length_side = sqrt(1 + (math.tan(theta))**2)
        
        
    A =  1 - length_side**(-2*gamma)
    # solve OT equation 
    R = symbols('R')
    expr = R**(2*(1-gamma)) - s**2
    
#    t0 = time.time()
    sol = sympy.solve(expr)
#    t = time.time() - t0
#    tot_time +=t
#    print('time for solving equation: ', t)
    R = sol[0]
    
    mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])
        

X = FunctionSpace(mesh_c,'CG',1)
x_OT = Function(X)
y_OT = Function(X)
    
v_d = dof_to_vertex_map(X)
      
x_OT.vector()[:] = mesh_OT.coordinates()[v_d,0]
y_OT.vector()[:] = mesh_OT.coordinates()[v_d,1]
    
Q = skewness(mesh_c,x_OT,y_OT)
Q_scalar = np.max(Q.vector()[:])   
print(Q_scalar)
np.save('Data/Q' + str(omega) + '.npy',Q_scalar)
#    
#string_mesh = 'mesh_OT/mesh_' + str(omega) + str(N) + '.xml.gz'
#File(string_mesh) << mesh_OT
#plot(mesh_OT)
#
#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu




    
