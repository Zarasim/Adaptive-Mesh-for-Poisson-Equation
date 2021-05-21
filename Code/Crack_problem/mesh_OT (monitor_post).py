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
import math
from quality_measure import *


parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.

def Newton(coeff,s,x,eps):
    
    A = coeff[0]
    B = coeff[1]
    gamma = coeff[2]
    
    f_value =  0.5*A*x**2 + B/(1-gamma)*x**(2*(1-gamma)) - 0.5*s**2
    dfdx = A*x + B*(2*(1-gamma))/(1-gamma)*x**(2*(1-gamma)-1)
    
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 100:
        
        try:
            x = x - float(f_value)/dfdx
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error

        f_value = 0.5*A*x**2 + B/(1-gamma)*x**(2*(1-gamma)) - 0.5*s**2
        dfdx = A*x + B*(2*(1-gamma))/(1-gamma)*x**(2*(1-gamma)-1)    
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
    
    
    return x, iteration_counter


File_q = File('Paraview/q.pvd')
File_mu = File('Paraview/mu.pvd')

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


q_vec = np.zeros(n_ref+1)
Q_vec = np.zeros(n_ref+1)
mu_vec = np.zeros(n_ref+1)
dof = np.zeros(n_ref+1)

for i in range(coords.shape[0]):
    
    print('iteration n°: ',i)
    # for each mesh point calculate the distance r     
    x = coords[i,:]
    #y = coords[i,1]
    s = np.sqrt(x[0]**2 + x[1]**2)
    
    theta = math.atan2(abs(x[1]),abs(x[0]))
        
    if (s==0):
        continue
      

    #A = abs(coeff[0])*1e5
    gamma = -coeff[2]/2
    B = coeff[1]*1e25

    # Find A by imposing boundary conditions
    #A = 1 - length_side**(-2*gamma)
    A = abs(coeff[0])*1e25
    coeff_ = [A,B,gamma]
    sol,it_counter = Newton(coeff_,s,0.1,eps=1e-12)
    R = sol
    
    mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])
   
V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
  
q = mesh_condition(mesh_OT)
mu = shape_regularity(mesh_OT)
q_vec[0] = np.max(q.vector()[:])
mu_vec[0] = np.min(mu.vector()[:])   
dof[0] = V.dim()


File_mu << mu
File_q << q



for it in range(1,n_ref+1):
 
  print('iteration n° ',it)
  mesh_OT = refine(mesh_OT) 
  V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
  
  q = mesh_condition(mesh_OT)
  mu = shape_regularity(mesh_OT)
  q_vec[it] = np.max(q.vector()[:])
  mu_vec[it] = np.min(mu.vector()[:])   
  dof[it] = V.dim()



    
