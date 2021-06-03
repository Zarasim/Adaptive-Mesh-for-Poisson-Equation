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
#from sympy import symbols
#from sympy import solve as sympsolve

import pandas as pd

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


tol = 1e-12
n_ref = 5

File_q = File('Paraview/q.pvd')
File_mu = File('Paraview/mu.pvd')


#gamma_vec = np.linspace(0.0,1.2,10)
gamma_vec = np.array([1.33])

for gamma in gamma_vec: 
            
    mesh_OT = Mesh('ell_mesh.xml')
    
    mesh_OT.rotate(-90)
    mesh_OT.coordinates()[:] = mesh_OT.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])
    coords = mesh_OT.coordinates()[:] 
    nv = coords.shape[0]

    
    print('Iteration for gamma: ', gamma)
    q_vec = np.zeros(n_ref+1)
    Q_vec = np.zeros(n_ref+1)
    mu_vec = np.zeros(n_ref+1)
    dof = np.zeros(n_ref+1)
    
    for i in range(coords.shape[0]):
        
         # for each mesh point calculate the distance r     
        x = coords[i]
        #y = coords[i,1]
        s = np.sqrt(x[0]**2 + x[1]**2)
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        # x[0] = 1
        if (theta >= 0) and (theta <= pi/4):
            length_side = sqrt(1 + math.tan(theta)**2)
        # x[1] = 1
        elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        # x[0] = -1
        elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
            length_side = sqrt(1 + abs(math.tan(theta))**2)
        # x[1] = -1 
        elif (theta > 5.0/4.0*pi) and (theta <= 3.0/2.0*pi):
            length_side = sqrt(1 + (1/abs(math.tan(theta)))**2)
        
        if (s==0):
            continue
        
        # Find alpha and beta to match the Lshaped boundary 
        # Fix beta to 1/3
        B = 1-gamma
        A = 1 - length_side**(-2*gamma)
       
        
#        R = symbols('R')
#        expr = A*R**2 + R**(2.0*(1-gamma)) - s**2
#        sol = sympsolve(expr)
#        
#        R = sol[0]   
        coeff_ = [A,B,gamma]
        sol,it_counter = Newton(coeff_,s,1-5,eps=1e-12)
        R = sol
        
        mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])
    
    plot(mesh_OT)   
    V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
    string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(gamma) + '_' + str(nv) + '.xml.gz'
    File(string_mesh) << mesh_OT  
    q = mesh_condition(mesh_OT)
    mu = shape_regularity(mesh_OT)
    q_vec[0] = np.max(q.vector()[:])
    mu_vec[0] = np.min(mu.vector()[:])   
    dof[0] = V.dim()
    
    #File_mu << mu
    #File_q << q
    
#    string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(gamma) + '_' + str(nv) + '.xml.gz'
#    File(string_mesh) << mesh_OT 
    
    for it in range(1,n_ref+1):
     
      print('iteration refinement n° ',it)
      mesh_OT = refine(mesh_OT) 
      V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
      
      nv = mesh_OT.coordinates()[:].shape[0]

      q = mesh_condition(mesh_OT)
      mu = shape_regularity(mesh_OT)
      q_vec[it] = np.max(q.vector()[:])
      mu_vec[it] = np.min(mu.vector()[:])   
      dof[it] = V.dim()
      string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(gamma) + '_' + str(nv) + '.xml.gz'
      File(string_mesh) << mesh_OT 
      
    
      

    
    #DG0 = FunctionSpace(mesh_OT,'DG',0)
    #mu = quality_measure(mesh_OT)
    #File('mu_OT' + str(N) + '.pvd') << mu
    
    
      np.save('Data/OT/a_priori/q/q_'+ str(gamma) +'.npy',q_vec)
      np.save('Data/OT/a_priori/mu/mu_' + str(gamma)  +'.npy',mu_vec)
      
      dict = {'dof': dof, 'mu': mu_vec, 'q': q_vec}  
       
      df = pd.DataFrame(dict) 
    
      # saving the dataframe 
      df.to_csv('Data/OT/a_priori/stat_' + str(np.round(gamma, 3)) + '.csv',index=False) 
