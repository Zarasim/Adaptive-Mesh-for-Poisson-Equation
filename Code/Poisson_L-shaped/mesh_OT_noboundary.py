#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone
    """

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from quality_measure import *

import pandas as pd

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


def Newton(coeff,s,x,eps,w=0.8):
    
    A = coeff[0]
    B = coeff[1]
    gamma = coeff[2]
    
    f_value = A*x**2 + B/(1-gamma)*x**(2*(1-gamma)) - s**2
    dfdx = 2*A*x + B/(1-gamma)*(2*(1-gamma))*x**(2*(1-gamma)-1)
    x_prev = x
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 2000:
        
        try:
            x = w*x_prev  + (1-w)*(x - float(f_value)/dfdx)
            x_prev = x
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error
        
        #print('it counter: ',it_counter)
        f_value = A*x**2 + B/(1-gamma)*x**(2*(1-gamma)) - s**2
        dfdx = 2*A*x + B/(1-gamma)*(2*(1-gamma))*x**(2*(1-gamma)-1)
        #print(dfdx)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
        print('convergence not reached')
    
    return x, iteration_counter



def generateOTmesh(mesh_uniform,mesh_OT):
    
    coords = mesh_uniform.coordinates()[:] 
    nVertices = coords.shape[0]
    
    for i in range(nVertices):
        
        # for each mesh point calculate the distance r     
        x = coords[i]
        s = np.sqrt(x[0]**2 + x[1]**2)
        
        if (s==0):
            continue
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        # side x[0] = 1
        if (theta >= 0) and (theta <= pi/4):
            length_side = sqrt(1 + math.tan(theta)**2)
        # side x[1] = 1
        elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        # side x[0] = -1
        elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
            length_side = sqrt(1 + abs(math.tan(theta))**2)
        # side x[1] = -1 
        elif (theta > 5.0/4.0*pi) and (theta <= 3.0/2.0*pi):
            length_side = sqrt(1 + (1/abs(math.tan(theta)))**2)
        
        # Find alpha and beta to match the Lshaped boundary 
        B = 1-gamma
        A = 1-length_side**(-2*gamma)
        
        # reduce initial condition for increasing gamma
        # 0.01 for gamma = 0.67
        # 1e-5 for gamma = 0.75-0.8
        # 1e-8 for gamma = 0.9
        
        coeff = [A,B,gamma]
        R,it_counter = Newton(coeff,s,1e-8,eps=1e-12)
        mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])  

        return mesh_OT


n_ref = 6
#gamma_vec = np.array([0.0,0.1,0.2,1.0/3.0,0.5,2.0/3.0,0.8,0.9])
gamma_vec = np.array([0.44])
output = 1


for gamma in gamma_vec: 
    print('Iteration for gamma: ', gamma)    
    
    mesh_uniform = Mesh('ell_mesh.xml')
    mesh_uniform.rotate(-90)
    mesh_uniform.coordinates()[:] = mesh_uniform.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])
    
    # create deep copy of mesh uniform
    mesh_OT = Mesh(mesh_uniform)
         
    if output:
        File_q = File('Paraview/OT_priori/q'+ str(np.round(gamma, 2)) + '.pvd')
        File_mu = File('Paraview/OT_priori/mu'+ str(np.round(gamma, 2)) + '.pvd')
    
   
    coords_uniform = mesh_uniform.coordinates()[:]     
    nVertices = coords_uniform.shape[0]

    q_vec = np.zeros(n_ref+1)
    mu_vec = np.zeros(n_ref+1)
    dof = np.zeros(n_ref+1)
    
    for i in range(nVertices):
        
        # for each mesh point calculate the distance r     
        x = coords_uniform[i]
        s = np.sqrt(x[0]**2 + x[1]**2)
        
        if (s==0):
            continue
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        # side x[0] = 1
        if (theta >= 0) and (theta <= pi/4):
            length_side = sqrt(1 + math.tan(theta)**2)
        # side x[1] = 1
        elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        # side x[0] = -1
        elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
            length_side = sqrt(1 + abs(math.tan(theta))**2)
        # side x[1] = -1 
        elif (theta > 5.0/4.0*pi) and (theta <= 3.0/2.0*pi):
            length_side = sqrt(1 + (1/abs(math.tan(theta)))**2)
        
        # Find alpha and beta to match the Lshaped boundary 
        B = 1-gamma
        A = 1-length_side**(-2*gamma)
        
        # reduce initial condition for increasing gamma
        # 0.01 for gamma = 0.67
        # 1e-5 for gamma = 0.75-0.8
        # 1e-8 for gamma = 0.9
        
        coeff = [A,B,gamma]
        R,it_counter = Newton(coeff,s,1e-8,eps=1e-12)
        mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])  
 
        # Approach by Chris
#        if(abs(x[1]) < abs(x[0])) and (x[1] != 0):
#            s_max = s/abs(x[1])
#        elif(x[0] != 0):
#            s_max = s/abs(x[0])
#        
#        print('Second run of Newton iteration')
#        R_max,it_counter = Newton(coeff,s_max,1e-2,eps=1e-12)
#        mesh_OT.coordinates()[i,:] = np.array([(R/R_max)*(s_max/s)*x[0],(R/R_max)*(s_max/s)*x[1]])
        

    V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
    q = mesh_condition(mesh_OT)
    mu = shape_regularity(mesh_OT)
    q_vec[0] = np.max(q.vector()[:])
    mu_vec[0] = np.min(mu.vector()[:])   
    dof[0] = V.dim()    
    string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(np.round(gamma, 2)) + '_' + str(nVertices) + '.xml.gz'
    File(string_mesh) << mesh_OT 
    
    
    if output:
        File_q << q,0
        File_mu << mu,0
        
    for it in range(1,n_ref+1):
     
      print(' refinement n° ',it)
      
      # first refine uniform mesh and then apply OT mesh 
      mesh_uniform = refine(mesh_uniform)      
      mesh_OT = Mesh(mesh_uniform)
      
      coords_uniform = mesh_uniform.coordinates()[:]
      nVertices = coords_uniform.shape[0]
      
      for i in range(nVertices):
        
        # for each mesh point calculate the distance r     
        x = coords_uniform[i]
        s = np.sqrt(x[0]**2 + x[1]**2)
        
        if (s==0):
            continue
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        # side x[0] = 1
        if (theta >= 0) and (theta <= pi/4):
            length_side = sqrt(1 + math.tan(theta)**2)
        # side x[1] = 1
        elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        # side x[0] = -1
        elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
            length_side = sqrt(1 + abs(math.tan(theta))**2)
        # side x[1] = -1 
        elif (theta > 5.0/4.0*pi) and (theta <= 3.0/2.0*pi):
            length_side = sqrt(1 + (1/abs(math.tan(theta)))**2)
        
        # Find alpha and beta to match the Lshaped boundary 
        B = 1-gamma
        A = 1-length_side**(-2*gamma)
        
        # reduce initial condition for increasing gamma
        # 0.01 for gamma = 0.67
        # 1e-5 for gamma = 0.75-0.8
        # 1e-8 for gamma = 0.9
        
        coeff = [A,B,gamma]
        R,it_counter = Newton(coeff,s,1e-8,eps=1e-12)
        mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])   
                  
      
      V = FunctionSpace(mesh_OT, "DG", 1) # function space for solution u     
      
      q = mesh_condition(mesh_OT)
      mu = shape_regularity(mesh_OT)
      
      q_vec[it] = np.max(q.vector()[:])
      mu_vec[it] = np.min(mu.vector()[:])   
      dof[it] = V.dim()
      
      string_mesh = 'Data/mesh/mesh_OT_priori/mesh_OT_' + str(np.round(gamma, 2)) + '_' + str(nVertices) + '.xml.gz'
      File(string_mesh) << mesh_OT 
      
      if output:
          File_q << q,it
          File_mu << mu,it
        
      
      np.save('Data/OT/a_priori/q/q_'+ str(np.round(gamma, 2)) +'.npy',q_vec)
      np.save('Data/OT/a_priori/mu/mu_' + str(np.round(gamma, 2))  +'.npy',mu_vec)
      
      dict = {'dof': dof, 'mu': mu_vec, 'q': q_vec}  
      df = pd.DataFrame(dict) 
      df.to_csv('Data/OT/a_priori/stat_' + str(np.round(gamma, 2)) + '.csv',index=False) 
