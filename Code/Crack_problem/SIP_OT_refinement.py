#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

@author: simo94
"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt 
import math
from quality_measure import *
import mshr
#import os
#
#os.getenv("HOME")
#
#data_path = '/home/simo94/PhD/SIP_method'
#pathset = os.path.join(data_path)

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

output = 0
n_ref = 5

eps = 0.001
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
mesh_c = mshr.generate_mesh(geometry, 10) 


## Solve Poisson Equation
Energy_norm = np.zeros(n_ref)
L2_norm = np.zeros(n_ref)
dof = np.zeros(n_ref)
q_vec = np.zeros(n_ref)
mu_vec = np.zeros(n_ref)
Q_vec = np.zeros(n_ref)


tol = 1e-12
f = Constant('0.0')

if output:
    file_u = File('Paraview/OT_ref/u.pvd')
    file_q = File('Paraview/OT_ref/q.pvd')
    file_mu = File('Paraview/OT_ref/mu.pvd')

# 236, 545, 2077, 8073
nvertices = 236
for it in range(n_ref):
 
  print('iteration n° ',it)
  
  if it == 0:
      string_mesh = 'mesh_OT/mesh_OT_' + str(nvertices) + '.xml.gz'
      mesh = Mesh(string_mesh)    
  else:
      mesh = refine(mesh)
      mesh_c = refine(mesh_c)
    

  DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
  V = FunctionSpace(mesh, "DG", 1) # function space for solution u

  u_exp = Expression_u(omega,degree=5)
#   
#  u = solve_poisson(u_exp)
#  mesh.bounding_box_tree().build(mesh)
  
  q = mesh_condition(mesh)
  mu = shape_regularity(mesh)
  
  if output:
      u.rename('u','u')
      mu.rename('mu','mu')
      q.rename('q','q')
      
      file_mu << mu,it
      file_q << q,it
      file_u << u 
      
  X = FunctionSpace(mesh_c,'CG',1)
  x_OT = Function(X)
  y_OT = Function(X)
      
  v_d = dof_to_vertex_map(X)
    
  x_OT.vector()[:] = mesh.coordinates()[v_d,0]
  y_OT.vector()[:] = mesh.coordinates()[v_d,1]    
  
  Q = skewness(mesh_c,mesh,x_OT,y_OT)
  #L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh))) 
  q_vec[it] = np.max(q.vector()[:])
  mu_vec[it] = np.min(mu.vector()[:])   
  Q_vec[it] = np.max(Q.vector()[:])
  dof[it] = V.dim()
#
##
#rate = conv_rate(dof,L2_norm)
###
fig, ax = plt.subplots()
ax.plot(dof,Q_vec,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %np.mean(rate[-1]))
ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')       
###
#
#np.save('Data/OT_ref/L2_OT.npy',L2_norm)
#np.save('Data/OT_ref/dof_OT.npy',dof)
#np.save('Data/OT_ref/rate_OT_L2.npy',rate)
#np.save('Data/OT_ref/q.npy',q_vec)
#np.save('Data/OT_ref/mu.npy',mu_vec)
#

