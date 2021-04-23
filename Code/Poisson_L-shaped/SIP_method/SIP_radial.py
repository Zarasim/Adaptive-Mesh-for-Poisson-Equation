#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

@author: simo94
"""


from dolfin import *
from supermesh import *
import numpy as np
import matplotlib.pyplot as plt
import mshr 
import math

import os

os.getenv("HOME")

data_path = '/home/simo94/PhD/SIP_method'
pathset = os.path.join(data_path)

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


# Expression for exact solution 
class Expression_u(UserExpression):
    
    def __init__(self,omega,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.omega = omega
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        elif x[0] > 0 and x[1] < 0:
            theta = 2*pi-theta
            
        if r == 0.0:
            value[0] = 0.0
        else:
            value[0] = pow(r,pi/self.omega)*sin(theta*pi/self.omega)
        
    def value_shape(self):
        return ()



def solve_poisson(u_exp):
    
    n = FacetNormal(mesh)
    h =  CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2
     
    # Define parameters
    #C_sigma = 20
    k_penalty = 20
    
    v = TestFunction(V)
    u = TrialFunction(V)
    
    # Define variational problem
    a = dot(grad(v), grad(u))*dx(mesh) \
       - dot(avg(grad(v)), jump(u, n))*dS(mesh) \
       - dot(jump(v, n), avg(grad(u)))*dS(mesh) \
       + (k_penalty/h_avg)*dot(jump(v, n), jump(u, n))*dS(mesh) \
       - dot(grad(v), u*n)*ds(mesh) \
       - dot(v*n, grad(u))*ds(mesh) \
       + (k_penalty/h)*v*u*ds(mesh)
    
    L = v*f*dx(mesh) - u_exp*dot(grad(v), n)*ds(mesh) + (k_penalty/h)*u_exp*v*ds(mesh)
 
    u = Function(V)
    
    solve(a==L,u)
    
    return u    



def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/(ln(dof[i+1]/dof[i]))

    return rate

output = 1



## Solve Poisson Equation
Energy_norm = np.zeros(5)
L2_norm = np.zeros(5)
dof = np.zeros(5)
  
## Pvd file


nvertices = np.array([136,384,1564,4834,22354])

it = 0
for nv in nvertices:

  print('iteration n° ',it)
  string_mesh = 'mesh_radial/my_mesh' + str(nv) + '.xml.gz'
  mesh = Mesh(string_mesh)

  if output:
    file_u = File('Paraview_radial/u_' + str(nv) +'.pvd')

 
  DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
  V = FunctionSpace(mesh, "DG", 1) # function space for solution u

  omega = 3.0/2.0*pi
  u_exp = Expression_u(omega,degree=5)
  f = Constant('0.0')
   
  u = solve_poisson(u_exp)
    
  if output:
      u.rename('u','u')
      file_u << u 
      
  mesh.bounding_box_tree().build(mesh)
  
  L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh))) 
  dof[it] = V.dim()
    
  it += 1      
  
rate = conv_rate(dof,L2_norm)


fig, ax = plt.subplots()
ax.plot(dof,L2_norm,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %np.mean(rate[-1]))
ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')       


np.save('Data/radial/L2_radial.npy',L2_norm)
np.save('Data/radial/dof_radial.npy',dof)
np.save('Data/radial/rate_radial_L2',rate)

