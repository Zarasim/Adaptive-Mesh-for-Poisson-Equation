#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone
    """

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pandas as pd
import mshr

from dolfin import *
from quality_measure import *

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
    
    while abs(f_value) > eps and iteration_counter < 500:
        
        try:
            x = w*x_prev  + (1-w)*(x - float(f_value)/dfdx)
            x_prev = x
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error
        
        #print('it counter: ',it_counter)
        f_value = A*x**2 + B/(1-gamma)*x**(2*(1-gamma)) - s**2
        dfdx = 2*A*x + B/(1-gamma)*(2*(1-gamma))*x**(2*(1-gamma)-1)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
        print('convergence not reached')
    
    return x, iteration_counter


output = 0

num = 10

# endpoint is included
#gamma_vec = np.linspace(0.0,0.74,num)

gamma_vec = np.array([0.6])
# fix refinement for different gammas
n_ref = 5


for gamma in gamma_vec: 
    
    print('Iteration for gamma: ', gamma)    
    
    N = 10
    omega =  0.5*pi/(1 - gamma)

    domain_vertices = [Point(0.0, 0.0),Point(1.0, 0.0)]
    
    if omega <= pi/4.0:
        domain_vertices.append(Point(1.0,np.tan(omega)))
    
    elif omega <= 3.0/4.0*pi:
        domain_vertices.append(Point(1.0,1.0))
        domain_vertices.append(Point(np.tan(pi/2 - omega),1.0))
    
    
    elif omega <= 5.0/4.0*pi:
        domain_vertices.append(Point(1.0,1.0))
        domain_vertices.append(Point(-1.0,1.0))
        domain_vertices.append(Point(-1,np.tan(pi-omega)))
    
    elif omega <= 7.0/4.0*pi:    
        domain_vertices.append(Point(1.0,1.0))
        domain_vertices.append(Point(-1.0,1.0))
        domain_vertices.append(Point(-1.0,-1.0))
        domain_vertices.append(Point(np.tan(omega - 3.0/2.0*pi), -1.0))

    elif omega > 7.0/4.0*pi:
        domain_vertices.append(Point(1.0,1.0))
        domain_vertices.append(Point(-1.0,1.0))
        domain_vertices.append(Point(-1.0,-1.0))                
        domain_vertices.append(Point(1.0, -1.0))
        domain_vertices.append(Point(1.0, -np.tan(2.0*pi - omega)))
   
        
    geometry = mshr.Polygon(domain_vertices)
    mesh_uniform = mshr.generate_mesh(geometry, N) 

    dof= np.zeros(n_ref)
    Q_vec = np.zeros(n_ref)
        
    for it,ref in enumerate(range(n_ref)):
 
        print(' Refinement n° ',ref+1)
      
        #first refine uniform mesh and then apply OT mesh 
        mesh_uniform = refine(mesh_uniform)        
        V = FunctionSpace(mesh_uniform, "DG", 1) # function space for solution u
        dof[it] = V.dim()   
        
       
        coords_uniform = mesh_uniform.coordinates()[:]
        nVertices = coords_uniform.shape[0]
        
        # create deep copy of uniform mesh
        mesh_OT = Mesh(mesh_uniform)
          
        if output:
            File_q = File('Paraview/OT_priori/q_'+ str(np.round(gamma, 2)) + '.pvd')
            File_mu = File('Paraview/OT_priori/mu_'+ str(np.round(gamma, 2)) + '.pvd')
            File_Q = File('Paraview/OT_priori/Q_'+ str(np.round(gamma, 2)) + '.pvd')
    
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
            
            coeff = [A,B,gamma]
            R,it_counter = Newton(coeff,s,1e-8,eps=1e-12)
            mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])
            
        plt.figure()
        plot(mesh_OT)
        V_OT = FunctionSpace(mesh_OT, "CG", 1)
        V_uniform = FunctionSpace(mesh_uniform, "CG", 1) 
        coords_P = V_OT.tabulate_dof_coordinates()
        
        x = Function(V_OT)
        y = Function(V_OT)
        x.vector()[:] = coords_P[:,0]
        y.vector()[:] = coords_P[:,1]
        
        Qs = skewness(mesh_uniform,x,y)
        coords_val = []
        iteration = 0
        
        for x,y in zip(coords_P[:,0],coords_P[:,1]):
              if x**2 + y**2 < (10**(-it))**2:
                 coords_val.append(iteration)
              iteration += 1   
        
        Q = Qs.vector()[coords_val]
        Q_vec[it] = np.max(Q)
        
        if output:
            File_Q << Q,it
        
    np.save('Data/Q_gamma' + str(round(gamma,2)) + '.npy',Q_vec)
#    dict = {'dof': dof, 'Q': Q_vec}
#    df = pd.DataFrame(dict) 
#    df.to_csv('Data/skewness' + str(round(gamma,2)) +'.csv',index=False)

