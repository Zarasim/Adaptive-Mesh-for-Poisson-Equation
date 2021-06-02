#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

Solve Poisson equation in region with corner singularity 

Model problem discussed in
Kim, Seokchan & Lee, Hyung-Chun. (2016). 
A finite element method for computing accurate solutions for Poisson equations with corner singularities using 
the stress intensity factor. Computers & Mathematics with Applications. 71. 10.1016/j.camwa.2015.12.023. 


@author: simo94
"""


## import packages
import numpy as np
import math 

import matplotlib.pyplot as plt
import mshr 

from dolfin import *

# Expression for exact solution 
class Expression_uexact(UserExpression):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs) # This part is new!
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        if r == 0.0:
            value[0] = 0.0
        else:
            value[0] = pow(r,2.0/3.0)*sin(2.0*theta/3.0)
        
    def value_shape(self):
        return ()
  
    
# Expression for exact solution 
class Expression_grad_uexact(UserExpression):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs) # This part is new!
    
    def eval(self, value, x):
        
        r = sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
        
        if r == 0.0:    
            value[0] = 0.0
            value[1] = 0.0
        else:
            
            #dr_u = (2.0/3.0)*np.power(r,-1.0/3.0)*sin(2*theta/3)
            #dtheta_u = (2.0/3.0)*np.power(r,-1.0/3.0)*cos(2*theta/3)
            
            #value[0] = cos(theta)*dr_u - sin(theta)*dtheta_u
            #value[1] = sin(theta)*dr_u + cos(theta)*dtheta_u
            value[0] = -(2.0/3.0)*pow(r,-1.0/3.0)*sin(1.0/3.0*theta)
            value[1] = (2.0/3.0)*pow(r,-1.0/3.0)*cos(1.0/3.0*theta)
            
    def value_shape(self):
        return (2,)    
    

def solve_poisson(u_expr):
    
    n = FacetNormal(mesh)
    
    v = TestFunction(CG1)
    u = TrialFunction(CG1)
    
    bcs = DirichletBC(CG1, u_expr,'on_boundary')
    
    # Weak form 
    a = inner(grad(v),grad(u))*dx(domain = mesh)
    L = Constant(0.0)*v*dx(domain = mesh)
    
    u = Function(CG1)
    solve(a==L,u,bcs)
    
    return u

def refinement(mesh,ref_ratio = False,tol=1.0):
    
    dx = Measure('dx',domain = mesh)
    dS = Measure('dS',domain = mesh)

    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    w = TestFunction(DG0)
    
    cell_residual = Function(DG0)

    # assume avg diameter close to edge length
    residual = h**2*w*(div(grad(u)))**2*dx + avg(w)*avg(h)*jump(grad(u),n)**2*dS
    assemble(residual, tensor=cell_residual.vector())

    # Compute tol
    theta = sum(cell_residual.vector()[:])/(mesh.num_cells())
    
    # Mark cells for refinement
    cell_markers = MeshFunction('bool',mesh,mesh.topology().dim())   

    if ref_ratio:
        
        gamma_0 = sorted(cell_residual.vector()[:],reverse = True)[int(mesh.num_cells()*ref_ratio)]
        gamma_0 = MPI.max(mesh.mpi_comm(),gamma_0)

        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[c.index()] > gamma_0
        
    else:
        # Apply equidistribution 
        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[c.index()] > theta*tol

    # Refine mesh 
    mesh = refine(mesh,cell_markers)
    
    return mesh


def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/(ln(sqrt(dof[i+1]/dof[i])))

    return rate

output = 1   

rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2

mesh = mshr.generate_mesh(geometry, 8) 
mesh.bounding_box_tree().build(mesh)

## Solve Poisson Equation
n_ref = 15
Energy_norm = np.zeros(n_ref)
L2_norm = np.zeros(n_ref)
dof = np.zeros(n_ref)
  
## Pvd file

if output:
    
    file_u = File('Paraview_poisson_href/u.pvd')

it = 0

DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)


while it < n_ref:

  print('iteration n° ',it)
  
  if it > 0:
      mesh = refinement(mesh,ref_ratio = 0.1)
 
  DG0 = FunctionSpace(mesh, "DG", 0)
  CG1 = FunctionSpace(mesh, "CG", 1)
  
  u_expr = Expression_uexact(degree=5)
  gradu_expr = Expression_grad_uexact(degree=5)
   
  u = solve_poisson(u_expr)
    
  if output:
      u.rename('u','u')
      file_u << u 
      
  mesh.bounding_box_tree().build(mesh)
  
  Energy_norm[it] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 
  L2_norm[it] = np.sqrt(assemble((u_expr - u)*(u_expr - u)*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 
  dof[it] = CG1.dim()
    
  it += 1      
  
      
# For CG1 convergence rate in H1 semi-norm is expected to be 1 
rate = conv_rate(dof,L2_norm)
rate_H1 = conv_rate(dof,Energy_norm)


fig, ax = plt.subplots()
ax.plot(dof,Energy_norm,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %rate_H1[-1])
ax.set_xlabel('dof')
ax.set_ylabel('H10 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')       
    

fig, ax = plt.subplots()
ax.plot(dof,L2_norm,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %rate[-1])
ax.set_xlabel('dof')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')       

np.save('H10_href.npy',Energy_norm)
np.save('L2_href.npy',L2_norm)
np.save('dof_href.npy',dof)
np.save('rate_href',rate_H1)
np.save('rate_href_L2',rate)

