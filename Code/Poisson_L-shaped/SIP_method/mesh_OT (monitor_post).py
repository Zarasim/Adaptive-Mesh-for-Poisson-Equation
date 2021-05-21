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
import sys
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


def bisection(coeff,r, x_L, x_R, eps, return_x_list=False):
    
    f_L = f(x_L,coeff,r)
    f_R = f(x_R,coeff,r)
    
    if f_L*f_R > 0:
        print("Error! Function does not have opposite signs at interval endpoints!")
        sys.exit(1)
        
    x_M = float(x_L + x_R)/2.0
    f_M = f(x_M,coeff,r)
    iteration_counter = 1
    if return_x_list:
        x_list = []

    while abs(f_M) > eps:
        if f_L*f_M > 0:   # i.e. same sign
            x_L = x_M
            f_L = f_M
        else:
            x_R = x_M
        x_M = float(x_L + x_R)/2
        f_M = f(x_M,coeff,r)
        iteration_counter += 1
        if return_x_list:
            x_list.append(x_M)
    if return_x_list:
        return x_list, iteration_counter
    else:
        return x_M, iteration_counter


def f(x,coeff,r):
    
    A = coeff[0]
    B = coeff[1]
    exponent = coeff[2]
    
    return 0.5*A*x**2 + B/(exponent+2)*x**(exponent+2) - 0.5*r**2


mesh_OT = Mesh('ell_mesh.xml')

mesh_OT.rotate(-90)
mesh_OT.coordinates()[:] = mesh_OT.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])



coeff = np.load('Data/coeff.npy')
coords = mesh_OT.coordinates()[:] 

tol = 1e-12


n_ref = 5

q_vec = np.zeros(n_ref+1)
Q_vec = np.zeros(n_ref+1)
mu_vec = np.zeros(n_ref+1)
dof = np.zeros(n_ref+1)

File_q = File('Paraview/q.pvd')
File_mu = File('Paraview/mu.pvd')

for i in range(coords.shape[0]):
    
    # for each mesh point calculate the distance r     
    x = coords[i,:]
    #y= coords[i,1]
    s = np.sqrt(x[0]**2 + x[1]**2)
    
    theta = math.atan2(abs(x[1]),abs(x[0]))
        
    if (s==0):
        continue
      

    #A = abs(coeff[0])*1e5
    gamma = -coeff[2]/2
    B = coeff[1]*1e5

    # Find A by imposing boundary conditions
    #A = 1 - length_side**(-2*gamma)
    A = abs(coeff[0])*1e5
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


  
#string_mesh = 'mesh_uniform/mesh_uniform_' + str(N) + '.xml.gz'
#ile(string_mesh) << mesh_OT


#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu



    
