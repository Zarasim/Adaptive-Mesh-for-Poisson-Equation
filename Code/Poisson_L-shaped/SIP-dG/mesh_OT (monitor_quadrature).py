#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone
    """

from dolfin import *
from quality_measure import *

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from sympy import symbols
from sympy import solve as sympsolve
import sympy


from scipy.interpolate import CubicSpline
import pandas as pd

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


def Newton(s,x,eps):

#    expr = sumR - 0.5*s**2
#    f_value = expr.evalf(subs={R:x})
#    dfdx_expr = sympy.diff(expr,R)
#    dfdx = dfdx_expr.evalf(subs={R:x})

    f_value = float(spl(x)) - 0.5*s**2
    dfdx_expr = spl.derivative()
    dfdx = float(dfdx_expr(x))
    
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 200:
        
        try:
            x = x - float(f_value)/dfdx
            
#            f_value = expr.evalf(subs={R:x})
#            dfdx = dfdx_expr.evalf(subs={R:x})
            f_value = float(spl(x)) - 0.5*s**2
            dfdx_expr = spl.derivative()
            dfdx = float(dfdx_expr(x))        
            iteration_counter += 1
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error
       
    
# Here, either a solution is found, or too many iterations
#    if abs(f_value) > eps:
#    iteration_counter = -1
     
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

# need to return a symbolic expression with r 
# find eq of line for each interval and compute symbolic integral
# return sum of integrals as symbolic expression of r
# for this you need to use sympy
        
def trapezoidal(r,w):
       
    dr = r[1:] - r[:-1]
    dw = w[1:] - w[:-1]
    m = dw/dr
    b = w[:-1] - m*r[:-1]
    sumR = 0.0
    
    for i in range(len(dr)-1):
        # find equation of straight line for each dr
        sumR += ((m[i]*R**2 + b[i]*R) + (m[i+1]*R**2 + b[i+1]*R))*dr[i]/2
    
    return sumR

        
def getIntegrand(r_vec,w_vec):
       
    r_avg = (r_vec[1:] + r_vec[:-1])/2

    w = np.zeros(len(w_vec))
    w[:-1] = r_avg*w_vec[:-1]
    w[-1] = r_vec[-1]*w_vec[-1]
    
    return w

R = symbols('R')

mesh_OT = Mesh('ell_mesh.xml')
mesh_OT.rotate(-90)
mesh_OT.coordinates()[:] = mesh_OT.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])
coords = mesh_OT.coordinates()[:] 

tol = 1e-12

n_ref = 5

q_vec = np.zeros(n_ref+1)
Q_vec = np.zeros(n_ref+1)
mu_vec = np.zeros(n_ref+1)
dof = np.zeros(n_ref+1)

File_q = File('Paraview/q.pvd')
File_mu = File('Paraview/mu.pvd')

nv0 = 32
# 16,32,64,120
w_vec = np.load('Data/r-adaptive/fit_data/monitor_'+ str(nv0)  +'.npy')
r_vec = np.load('Data/r-adaptive/fit_data/dist_'+ str(nv0) +'.npy')

# get integrand value by multiplying by r
w_vec = getIntegrand(r_vec,w_vec)

# all for 16
# all for 32
# :30 or :110  :-1 for 64 
#     for 120


w = w_vec[:]
r = r_vec[:]


## integrate lhs with trapezium rule
#sumR = trapezoidal(r_vec,w_vec)

## spline interpolation 
spl = CubicSpline(r,w)
plt.loglog(r_vec,w_vec,'o',r, spl(r),'-')

for i in range(coords.shape[0]):
    
    # for each mesh point calculate the distance r   
    x = coords[i,:]
    s = np.sqrt(x[0]**2 + x[1]**2)   
        
    if (s==0):
        continue
    
    # Initial guess is critical for the convergence of the method 
    # keep s for N = 32
    
    sol,it_counter = Newton(s,s,eps=1e-12)
    R = sol
    mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])

    
mt = MeshTransformation
mt.rotate(mesh_OT,180,2,Point(0,0))
mesh_OT.coordinates()[:] = mesh_OT.coordinates()[:]*1e3


plt.figure()
plot(mesh_OT)
V = FunctionSpace(mesh_OT, "DG", 1)  # function space for solution u     
q = mesh_condition(mesh_OT)
mu = shape_regularity(mesh_OT)
q_vec[0] = np.max(q.vector()[:])
mu_vec[0] = np.min(mu.vector()[:])   
dof[0] = V.dim()
nv = mesh_OT.coordinates()[:].shape[0]

string_mesh = 'Data/mesh/mesh_OT_priori_quad/nv0_' + str(nv0) + '/mesh_OT_'+ str(nv) + '.xml.gz'
File(string_mesh) << mesh_OT 

File_mu << mu
File_q << q


for it in range(1,n_ref+1):
 
  print('iteration n° ',it)
  mesh_OT = refine(mesh_OT) 
  V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
  nv = mesh_OT.coordinates()[:].shape[0]
  q = mesh_condition(mesh_OT)
  mu = shape_regularity(mesh_OT)
  q_vec[it] = np.max(q.vector()[:])
  mu_vec[it] = np.min(mu.vector()[:])   
  dof[it] = V.dim()
  
  string_mesh = 'Data/mesh/mesh_OT_priori_quad/nv0_' + str(nv0) + '/mesh_OT_'+ str(nv) + '.xml.gz'
  File(string_mesh) << mesh_OT 

  np.save('Data/OT/a_priori_quad/q_' + str(nv) + '.npy',q_vec)
  np.save('Data/OT/a_priori_quad/mu_'+ str(nv) + '.npy',mu_vec)
      
dict = {'dof': dof, 'mu': mu_vec, 'q': q_vec}  
       
df = pd.DataFrame(dict) 
# saving the dataframe 
df.to_csv('Data/OT/a_priori_quad/stat_' + str(nv0) + '.csv',index=False) 

  

#DG0 = FunctionSpace(mesh_OT,'DG',0)
#mu = quality_measure(mesh_OT)
#File('mu_OT' + str(N) + '.pvd') << mu



    
