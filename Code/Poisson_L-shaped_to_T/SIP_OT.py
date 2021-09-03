#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

Import mesh from mesh_OT_local_2 with 5 re-entrant corners in corners list

Solve Poisson equation in solve_poisson() on the domain with assigned Dirichlet
Boundary condition in Expression_u.

    Assign 0 Boundary condition everywhere or not ?
    What is the exact solution on the new domain ?
    Use local a-posteriori error estimate to check for equidistribution ?

@author: simo94
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math

from dolfin import *
from quality_measure import *


parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


# Expression for exact solution 
class Expression_u(UserExpression):
    
    '''
        Expression for exact solution
    
    '''
    
    def __init__(self,omega,corners,**kwargs):
        super().__init__(**kwargs) 
        self.omega = omega
        self.corners = corners  # list of re-entrant corners
    
    # assign boundary conditions
    def eval(self, value, xv):
        
        # find closest corner from x and evaluate expression 
        r = 100
        #corner = [0.0,0.0]
        
        value[0] = 0.0
        
        for idx,corner in enumerate(self.corners):
            
            x = xv - corner   
            theta = math.atan2(abs(x[1]),abs(x[0]))
            
            if idx == 0:
                
                if x[0] < 0 and x[1] > 0:
                    theta = pi - theta
                    
                elif x[0] <= 0 and x[1] <= 0:
                    theta = pi + theta
                    
                if (r == 0.0): 
                #or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] >= 0):           
                    value[0] += 0.0
                else:
                    value[0] += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                
            elif idx == 1:
                
                if x[0] <= 0 and x[1] >= 0:
                    theta = pi/2 - theta
                
                elif x[0] <= 0 and x[1] <= 0:
                    theta = pi/2 + theta
                    
                elif x[0] > 0 and x[1] < 0:
                  theta = 3/2*pi-theta  
                
                if (r == 0.0): 
                #or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] >= 0):           
                    value[0] += 0.0
                else:
                    value[0] += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                    
                        
            elif idx == 2:
                
                if x[0] >= 0 and x[1] <= 0:
                    theta = pi - theta
                                
                elif x[0] >= 0 and x[1] >= 0:
                    theta = pi + theta
                              
                if (r == 0.0):
                #or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] <= 0):
                    value[0] += 0.0
                else:
                    value[0] += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                        
            
            else:
                
                 if x[0] >= 0 and x[1] <= 0:
                     theta = pi/2 - theta
            
                 elif x[0] > 0 and x[1] > 0:
                    theta = pi/2 + math.atan2(abs(x[1]),abs(x[0]))
                            
                 elif x[0] < 0 and x[1] > 0:
                    theta = 3*pi/2 - math.atan2(abs(x[1]),abs(x[0]))
                          
                 if (r == 0.0):#
                 #or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] <= 0):
                     value[0] += 0.0     
                 else:
                    value[0] += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                    
        
    # compute L-00 error 
    def eval_at_point(self, xv):
        
          # find closest corner from x and evaluate expression 
        r = 100
        #corner = [0.0,0.0]
        
        value = 0.0
        
        for idx,corner in enumerate(self.corners):
            
            x = xv - corner   
            theta = math.atan2(abs(x[1]),abs(x[0]))
            
            if idx == 0:
                
                if x[0] < 0 and x[1] > 0:
                    theta = pi - theta
                    
                elif x[0] <= 0 and x[1] <= 0:
                    theta = pi + theta
                    
                if (r == 0.0): 
                #or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] >= 0):           
                    value += 0.0
                else:
                    value += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                
            elif idx == 1:
                
                if x[0] <= 0 and x[1] >= 0:
                    theta = pi/2 - theta
                
                elif x[0] <= 0 and x[1] <= 0:
                    theta = pi/2 + theta
                    
                elif x[0] > 0 and x[1] < 0:
                  theta = 3/2*pi-theta  
                
                if (r == 0.0): 
                #or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] >= 0):           
                    value += 0.0
                else:
                    value += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                    
                        
            elif idx == 2:
                
                if x[0] >= 0 and x[1] <= 0:
                    theta = pi - theta
                                
                elif x[0] >= 0 and x[1] >= 0:
                    theta = pi + theta
                              
                if (r == 0.0):
                #or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] <= 0):
                    value += 0.0
                else:
                    value += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                        
            
            else:
                
                 if x[0] >= 0 and x[1] <= 0:
                     theta = pi/2 - theta
            
                 elif x[0] > 0 and x[1] > 0:
                    theta = pi/2 + math.atan2(abs(x[1]),abs(x[0]))
                            
                 elif x[0] < 0 and x[1] > 0:
                    theta = 3*pi/2 - math.atan2(abs(x[1]),abs(x[0]))
                          
                 if (r == 0.0):#
                 #or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] <= 0):
                     value += 0.0     
                 else:
                    value += pow(r,pi/self.omega)*sin(theta*pi/self.omega)
                    
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

class Expression_cell(UserExpression):
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        values[0] = cell.volume()                
        
    def value_shape(self):
        return ()


num = 8
gamma_vec = np.linspace(0.1,0.8,num)
#gamma_vec = np.array([0.5])
n_ref = 4

# Solve Poisson Equation
Linfty_norm = np.zeros(n_ref)


output = 1
corners = [[3.0,1.0],[2.0,4.0],[-6.0,5.0],[-5.0,3.0]]


for gamma in gamma_vec:
     
    print('Iteration for gamma° : ',gamma)
    L2_norm = np.zeros(n_ref)
    dofs = np.zeros(n_ref)
        
    for it in range(n_ref):
        
       # compute error for the OT mesh with fixed dof 
       print('Iteration for ref° : ', it)
       
       if output:
           file_u = File('Paraview/u_gamma_' + str(round(gamma,2)) + '_ref_' + str(it) + '.pvd')
           
           
       string_mesh = 'Mesh/' + str(round(gamma,2)) + '/' + 'mesh_T_' + str(it) + '.xml.gz'
       mesh = Mesh(string_mesh)    
       coords = mesh.coordinates()[:]
       
       DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
       DG1 = FunctionSpace(mesh, "DG", 1) 
       CG1 = FunctionSpace(mesh,"CG",3)
       V = FunctionSpace(mesh, "DG", 1) # function space for solution u
       dofs[it] = V.dim()
       
       omega = 3.0/2.0*pi
       u_exp = Expression_u(omega,corners,degree=3)
       uex = interpolate(u_exp,CG1)
       f = -div(grad(uex))
       u = solve_poisson(u_exp)

       mesh.bounding_box_tree().build(mesh)   
       
       if output:
          u.rename('u','u')    
          file_u << u,it
#              
#       L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh)))
#       
       maxErr = 0
       for i,x in enumerate(coords):
           err = abs(u_exp.eval_at_point(x) - u(x))
           if err > maxErr:
               maxErr = err
           
       Linfty_norm[it] = maxErr
            
    #np.save('Data/L2_'  + str(np.round(gamma, 2)) + '.npy',L2_norm)
    #np.save('Data/dofs_'  + str(np.round(gamma, 2)) + '.npy',dofs)
    np.save('Data/Linfty_'  + str(np.round(gamma, 2)) + '.npy',Linfty_norm)
    #dict_err = {'gamma': gamma_vec, 'error_L2': L2_norm}   
    #df = pd.DataFrame(dict_err) 
    #df.to_csv('Data/error_gamma.csv',index=False) 
