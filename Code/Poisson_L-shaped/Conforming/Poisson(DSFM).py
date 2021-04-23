#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

Solve Poisson equation in L-shaped domain using DSFM 

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
    
    def __init__(self,rho,omega,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.rho = rho
        self.omega = omega
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
                
        if r > 0 and r < 0.5*self.rho:
            
            value[0] = pow(r,pi/self.omega)*sin(pi*theta/self.omega)
        
        elif r >= 0.5*self.rho and r <= self.rho:
        
            p = 4*r/self.rho - 3
            value[0] = (15.0/16.0)*(8.0/15.0 - p + 2.0/3.0*pow(p,3)  - 1.0/5.0*pow(p,5))*pow(r,pi/self.omega)*sin(pi*theta/self.omega)
        
        else:
            
            value[0] = 0.0
        
    def value_shape(self):
        return ()

    
class Expression_eta(UserExpression):
    
    def __init__(self,rho,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.rho = rho
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
                
        if r > 0 and r < 0.5*self.rho:
            
            value[0] = 1.0
        
        elif r >= 0.5*self.rho and r <= self.rho:
        
            p = 4*r/self.rho - 3
            value[0] = (15.0/16.0)*(8.0/15.0 - p + 2.0/3.0*pow(p,3)  - 1.0/5.0*pow(p,5))
        
        else:
            
            value[0] = 0.0
        
    def value_shape(self):
        return ()    

    
# Expression for exact solution 
class Expression_s(UserExpression):
    
    def __init__(self,omega,plus,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.omega = omega
        self.plus = plus
        
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
        
        if r == 0:
            value[0] = 0.0
        else:
            value[0] = pow(r,self.plus*(pi/self.omega))*sin(pi*theta/self.omega)
        
    def value_shape(self):
        return ()
        
    
def solve_poisson():

    v = TestFunction(CG1)
    u = TrialFunction(CG1)
    
    bcs = DirichletBC(CG1, 0.0,'on_boundary')
    
    # Weak form 
    a = inner(grad(v),grad(u))*dx(domain = mesh)
    L = f*v*dx(domain = mesh)
    
    u = Function(CG1)
    solve(a==L,u,bcs)
    
    return u


def solve_regular_poisson(lambda_h):
    
    v = TestFunction(CG1)
    w = TrialFunction(CG1)
    
    bcs = DirichletBC(CG1,0.0,'on_boundary')
    
    # Weak form 
    a = inner(grad(v),grad(w))*dx(domain = mesh)
    L = f*v*dx(domain = mesh) + lambda_h*div(grad(eta*sp))*v*dx(domain=mesh)
    
    w = Function(CG1)
    solve(a==L,w,bcs)
    
    return w


tol = 1e-14
## Define domain boundaries
def boundary_1(x, on_boundary):
     return on_boundary and near(x[1], -1.0, tol)

def boundary_2(x, on_boundary):
     return on_boundary and near(x[0], 0.0, tol)

def boundary_3(x, on_boundary):
     return on_boundary and near(x[1], 0.0, tol)

def boundary_4(x, on_boundary):
     return on_boundary and near(x[0], 1.0, tol)

def boundary_5(x, on_boundary):
     return on_boundary and near(x[1], 1.0, tol)

def boundary_6(x, on_boundary):
     return on_boundary and near(x[0], -1.0, tol)


def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/ln(sqrt(dof[i+1]/dof[i]))

    return rate


Ns = 2**np.arange(3,10)
output = 0
#parameters['allow_extrapolation'] = True
H10err = np.zeros(Ns.shape[0])
L2err =  np.zeros(Ns.shape[0])
dof = np.zeros(Ns.shape[0])


## Create L-shaped domain 
rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2


for it,N in enumerate(Ns):
    
    mesh = mshr.generate_mesh(geometry, N)   # physical mesh 
    
    CG1 = FunctionSpace(mesh, "CG", 1)
    CG_exact = FunctionSpace(mesh, "CG", 3)
    
    u_expr = Expression_uexact(rho = 0.8,omega= (3.0/2.0)*pi,degree=3)
    sp_expr = Expression_s(omega= (3.0/2.0)*pi,plus = 1.0,degree=3)
    sm_expr = Expression_s(omega= (3.0/2.0)*pi,plus = -1.0,degree=3)
    eta_expr = Expression_eta(rho = 0.8,degree=3)
    
    u_exact = interpolate(u_expr,CG_exact)
    eta = interpolate(eta_expr,CG_exact) 
    sm = interpolate(sm_expr,CG_exact)
    sp = interpolate(sp_expr,CG_exact)
    
    f = -div(grad(u_exact))
    
    ## Solve Poisson Equation using standard FEM 
    u_h = solve_poisson()
    
    ## Compute stress intensity factor using extraction formula 
    lambda_h = 1/pi*assemble(f*eta_expr*sm_expr*dx(mesh)) + 1/pi*assemble(u_h*div(grad(eta*sm))*dx(mesh))
        
    ## Compute solution of new inhomogenous problem 
    w = solve_regular_poisson(lambda_h)
    
    # final solution 
    sp = interpolate(sp_expr,CG1)
    eta = interpolate(eta_expr,CG1)
    u = Function(CG1)
    
    u.vector()[:] = w.vector()[:] + lambda_h*eta.vector()[:]*sp.vector()[:]
    
    # compute error between exact and computationa solution in H10 nornm 
    H10err[it] = np.sqrt(assemble(dot(grad(u_exact) - grad(u),grad(u_exact) - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 3}))
    L2err[it] = np.sqrt(assemble((u_expr - u)*(u_expr - u)*dx(mesh),form_compiler_parameters = {"quadrature_degree": 3}))
    
    #H10err[it] = errornorm(u_expr,u,norm_type='H10',degree_rise=3,mesh=mesh) 
    #L2err[it] = errornorm(u_expr,u,norm_type='l2',degree_rise=3,mesh=mesh)
    dof[it] = CG1.dim()  


if output:
    
    file_u = File('Paraview_poisson_DSFM/u.pvd')
    file_uexact = File('Paraview_poisson_DSFM/u_exact.pvd')
    
    file_u << u 
    file_uexact << u_exact


rate_H10 = conv_rate(dof,H10err)
rate_L2 = conv_rate(dof,L2err)

plt.figure()
plt.plot(dof,H10err,linestyle = '-.', marker = 'o',label = 'rate: %4g' %rate_H10[-1])
plt.plot(dof,L2err,linestyle = '-.', marker = 'o',label = 'rate: %4g' %rate_L2[-1])
plt.xlabel('dof')
plt.ylabel('H10 error')
plt.xscale('log')
plt.yscale('log')
plt.legend()

np.save('H10_poisson(DSFM).npy',H10err)
np.save('L2_poisson(DSFM).npy',L2err)
np.save('rateH10_poisson(DSFM).npy',rate_H10)
np.save('rateL2_poisson(DSFM).npy',rate_L2)
np.save('dof(DSFM).npy',dof)
