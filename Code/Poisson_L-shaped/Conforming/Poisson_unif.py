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


parameters['allow_extrapolation'] = True


# Expression for exact solution 
class Expression_uexact(UserExpression):
    
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
  
    
# Expression for exact solution 
class Expression_grad_uexact(UserExpression):
    
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



def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/(ln(np.sqrt(dof[i+1]/dof[i])))
    
    return rate


## Create L-shaped domain 
output = 1

# set maximum of 9 or 10 
N = 2**np.arange(3,7)
#rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
#rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))


eps = 0.01
omega = 2*pi - eps

domain_vertices = [Point(0.0, 0.0),
                   Point(1.0, 0.0),
                   Point(1.0, 1.0),
                   Point(-1.0, 1.0),
                   Point(-1.0, -1.0)]

if omega - 3/2*pi < pi/4:    
    domain_vertices.append((np.tan(omega - 3/2*pi), -1.0))

else:
    
    alpha = 2*pi - omega
    domain_vertices.append(Point(1.0, -1.0))
    domain_vertices.append(Point(1.0, -np.tan(alpha)))


domain = mshr.Polygon(domain_vertices)

#geometry = rectangle1 + rectangle2

H10_norm = np.zeros(len(N))
L2_norm = np.zeros(len(N))
dof = np.zeros(len(N))


if output:
    
    file_u = File('Paraview_poisson_unif/u.pvd')
    file_uexact = File('Paraview_poisson_unif/u_exact.pvd')
 
mesh = mshr.generate_mesh(domain, 10)     

for it,Nv in enumerate(N):
       
    
    print('iteration n° ',it)
    
    if it > 0:
        mesh = refine(mesh)
    
    #mesh = mshr.generate_mesh(geometry, Nv) 
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    
    u_expr = Expression_uexact(omega,degree=5)
    gradu_expr = Expression_grad_uexact(omega,degree=5)
 
    ## Solve Poisson Equation
    u = solve_poisson(u_expr)
  
    mesh.bounding_box_tree().build(mesh)
    
    H10_norm[it] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 
    L2_norm[it] = np.sqrt(assemble((u_expr - u)*(u_expr - u)*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 
    
    dof[it] = CG1.dim()

    if output:
        u.rename('u','u')
        file_u << u 
     
    it += 1  

# For CG1 convergence rate in H1 semi-norm is expected to be 1 
rate = conv_rate(dof,L2_norm)
rate_H1 = conv_rate(dof,H10_norm)

fig, ax = plt.subplots()
ax.plot(dof,H10_norm,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %rate_H1[-1])
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
    
   

np.save('H10_unif.npy',H10_norm)
np.save('L2_unif.npy',L2_norm)
np.save('dof_unif.npy',dof)
np.save('rate_unif',rate_H10)
np.save('rate_unif_L2',rate_L2)
