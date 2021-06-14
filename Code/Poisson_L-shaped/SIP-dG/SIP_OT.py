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
import pandas as pd


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
            
    def eval_at_point(self, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        elif x[0] > 0 and x[1] < 0:
            theta = 2*pi-theta
            
        if r == 0.0:
            value = 0.0
        else:
            value = pow(r,pi/self.omega)*sin(theta*pi/self.omega)
            
        return value
        
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
    rate = np.zeros(l)
       
    rate[0] = 0
    for i in range(1,l-1):
        rate[i] = ln(err[i]/err[i+1])/(ln(dof[i+1]/dof[i]))

    return rate


#gamma_vec = np.array([0.0,0.1,0.2,1.0/3.0,0.5,2.0/3.0,0.8,0.9])
gamma_vec = np.array([0.8,0.9])

output = 1

for gamma in gamma_vec[:]:
    print('Iteration for gamma: ', gamma)
    nvertices = np.array([65,225,833,3201,12545,49665,197633])
    
    ## Solve Poisson Equation
    L2_norm = np.zeros(len(nvertices))
    Linfty_norm = np.zeros(len(nvertices))
    dof = np.zeros(len(nvertices))

    it = 0
    if output:
        file_u = File('Paraview/OT_priori/u.pvd')
    
    for it,nv in enumerate(nvertices):
       
       print(' refinement n° ',it)
    
       string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(np.round(gamma, 2)) + '_' + str(nv) + '.xml.gz'
       mesh = Mesh(string_mesh)    
       coords = mesh.coordinates()[:]
       
       DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
       DG1 = FunctionSpace(mesh, "DG", 1) 
       CG1 = FunctionSpace(mesh,"CG",1)
       V = FunctionSpace(mesh, "DG", 1) # function space for solution u
    
       omega = 3.0/2.0*pi
       u_exp = Expression_u(omega,degree=5)
       f = Constant('0.0')
       
       u = solve_poisson(u_exp)
       mesh.bounding_box_tree().build(mesh)   
       
       if output:
          u.rename('u','u')    
          file_u << u,it
       
        
       dof[it] = V.dim()
       L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh)))
       
       maxErr = 0
       for i,x in enumerate(coords):
           err = abs(u_exp.eval_at_point(x) - u(x))
           if err > maxErr:
               maxErr = err
           
       Linfty_norm[it] = maxErr
       
       it += 1      
      
    rate_L2 = conv_rate(dof,L2_norm)
    rate_Linfty = conv_rate(dof,Linfty_norm)
    

    np.save('Data/OT/a_priori/err/L2_'  + str(np.round(gamma, 2)) + '.npy',L2_norm)
    np.save('Data/OT/a_priori/err/Linfty_'  + str(np.round(gamma, 2)) + '.npy',Linfty_norm)
    np.save('Data/OT/a_priori/err/dof_' + str(np.round(gamma, 2)) +'.npy',dof)
    
    np.save('Data/OT/a_priori/err/rateL2_' + str(np.round(gamma, 2)) +'.npy',rate_L2)
    np.save('Data/OT/a_priori/err/rateLinfty_' + str(np.round(gamma, 2)) +'.npy',rate_Linfty)
    
    dict = {'dof': dof, 'error': L2_norm, 'rate': rate_L2}  
    df = pd.DataFrame(dict) 
    df.to_csv('Data/OT/a_priori/errorL2_' + str(np.round(gamma, 2)) + '.csv',index=False) 
    
    
    dict = {'dof': dof, 'error': Linfty_norm, 'rate': rate_Linfty}  
    df = pd.DataFrame(dict) 
    df.to_csv('Data/OT/a_priori/errorLinfty_' + str(np.round(gamma, 2)) + '.csv',index=False) 
    