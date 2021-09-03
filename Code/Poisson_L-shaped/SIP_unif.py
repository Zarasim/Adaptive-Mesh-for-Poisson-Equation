#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

@author: simo94
"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import mshr 
import math
from quality_measure import *
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



class Expression_grad(UserExpression):
    
    def __init__(self,omega,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.omega = omega
    
    def eval(self, value, x):
        
        r = sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
        
        elif x[0] > 0 and x[1] < 0:
            theta = 2*pi-theta
            
        if r == 0.0:    
            value[0] = 0.0
            value[1] = 0.0
        else:
            
            value[0] = -(pi/self.omega)*pow(r,pi/self.omega - 1)*sin((self.omega - pi)/self.omega*theta)
            value[1] = (pi/self.omega)*pow(r,pi/self.omega - 1)*cos((self.omega - pi)/self.omega*theta)
            
    def value_shape(self):
        return (2,)    
    

class Expression_aposteriori(UserExpression):
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        #n = cell.normal(ufc_cell.local_facet)
        
        if cell.contains(Point(0.0,0.0)):
            
           return np.float(1)
       
        else:
            
            return np.float(0.0)
        
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

rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2
#
mesh = mshr.generate_mesh(geometry, 10) 
mesh.bounding_box_tree().build(mesh)

### second option: import uniform mesh from xml file
#mesh = Mesh('ell_mesh.xml')
#mesh.rotate(-90)
#mesh.coordinates()[:] = mesh.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])


## Solve Poisson Equation
n_ref = 5
Energy_norm = np.zeros(n_ref)
L2_norm = np.zeros(n_ref)
dof = np.zeros(n_ref)
q_vec = np.zeros(n_ref)
mu_vec = np.zeros(n_ref)

## Pvd file

if output:
    file_mu = File('Paraview/Unif-ref/mu.pvd')
    file_q = File('Paraview/Unif-ref/q.pvd')

it = 0

while it < n_ref:

  print('iteration n° ',it)
  
  if it > 0:
      mesh = refine(mesh)
 
  DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
  V = FunctionSpace(mesh, "DG", 1) # function space for solution u

  omega = 3*pi/2
  u_exp = Expression_u(omega,degree=5)
  gradu_exp = Expression_grad(omega,degree=5)
  f = Constant('0.0')
   
  u = solve_poisson(u_exp)
    
  q = mesh_condition(mesh)
  mu = shape_regularity(mesh)
  
  if output:
      mu.rename('mu','mu')
      q.rename('q','q')
      
      file_mu << mu,it
      file_q << q,it
      
  mesh.bounding_box_tree().build(mesh)
  q_vec[it] = np.max(q.vector()[:])
  mu_vec[it] = np.min(mu.vector()[:])
  
#  L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh))) 
  dof[it] = V.dim()
    
  it += 1      
  
#rate = conv_rate(dof,L2_norm)
#,label = 'rate: %.4g' %np.mean(rate[-1])

fig, ax = plt.subplots()
ax.plot(dof,q_vec,linestyle = '-.',marker = 'o')
ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')       

#
#np.save('Data/unif/L2_unif.npy',L2_norm)
#np.save('Data/unif/dof_unif.npy',dof)
#np.save('Data/unif/rate_unif_L2',rate)
np.save('Data/unif_ref/q.npy',q_vec)
np.save('Data/unif_ref/mu.npy',mu_vec)