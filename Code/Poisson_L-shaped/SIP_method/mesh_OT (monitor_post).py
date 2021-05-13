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


def Newton(coeff,r,x,eps):
    
    A = coeff[0]*1e10
    B = coeff[1]*1e10
    exponent = coeff[2]
    
    f_value = 0.5*A*x**2 + B/(exponent+2)*x**(exponent+2) - 0.5*r**2
    dfdx = A*x + B*x**(exponent+1)
    
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 100:
        try:
            x = x - float(f_value)/dfdx
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error

        f_value = 0.5*A*x**2 + B/(exponent+2)*x**(exponent+2) - 0.5*r**2
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


coeff = np.load('Data/coeff/coeff.npy')
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
      
    #R, no_iterations = Newton(coeff,r, x=1e-3, eps=1.0e-6)
    #R, no_iterations = bisection(coeff,r,0,10^8, eps=1.0e-6)

    R = symbols('R')
    A = coeff[0]*1e10
    B = coeff[1]*1e10
    exponent = coeff[2]
    
    expr = 0.5*A*R**2 + B/(exponent+2)*R**(exponent+2) - 0.5*r**2
    t0 = time.time()
    sol = solve(expr)
    t = time.time() - t0
    tot_time +=t
    print('time for solving equation: ', t)
    R = sol[0]
    
    mesh_OT.coordinates()[i,:] = np.array([R*x/r,R*y/r])
    
print('total time elapsed: ',tot_time)
#string_mesh = 'mesh_uniform/mesh_uniform_' + str(N) + '.xml.gz'
#ile(string_mesh) << mesh_OT
plot(mesh_OT)

#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu



    
