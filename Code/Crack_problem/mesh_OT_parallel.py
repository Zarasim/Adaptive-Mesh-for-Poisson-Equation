#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone
"""

from dolfin import *
import numpy as np
import math
import matplotlib.pyplot as plt
import time 
from sympy import symbols
from sympy import solve as sympsolve
import multiprocessing as mp 
import mshr
from quality_measure import *


parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


def solveOT(idx_start,coords,beta):
    
    R_vec = np.zeros(len(coords))
    s_vec = np.zeros(len(coords))
    
    for i in range(coords.shape[0]):
        
        # for each mesh point calculate the distance r     
        x = coords[i]
        #y = coords[i,1]
        s = np.sqrt(x[0]**2 + x[1]**2)
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
        
        elif x[0] >= 0 and x[1] <= 0:
            theta = 2*pi - theta
            
        # x[0] = 1
        if (theta >= 0) and (theta <= pi/4):
            length_side = sqrt(1 + math.tan(theta)**2)
        # x[1] = 1
        elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        # x[0] = -1
        elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
            length_side = sqrt(1 + math.tan(theta)**2)
        # x[1] = -1 
        elif (theta > 5.0/4.0*pi) and (theta <= 7.0/4.0*pi):
            length_side = sqrt(1 + (1/math.tan(theta))**2)
        
        elif theta > 7.0/4.0*pi:
            length_side = sqrt(1 + (math.tan(theta))**2)
                
        
        if (s==0):
            continue
        
        # Find alpha and beta to match the Lshaped boundary 
        # beta <= 1/4
        #beta = 1.0/6.0
        #alpha = 1 - 4*beta*length_side**(-3.0/2.0)
        alpha = 1 - length_side**(-3.0/2.0)
        

        # solve OT equation 
        R = symbols('R')
        expr = alpha*R**2 + 4.0*beta*R**(1.0/2.0) - s**2
        sol = sympsolve(expr)
        
        R_vec[i] = sol[0]    
        #print(sol[0])
        s_vec[i] = s
        
        
    return (idx_start,R_vec,s_vec)



eps = 0.001
N = 10
beta = 1/4
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


mesh_c = mshr.generate_mesh(geometry, N) 
mesh_OT = mshr.generate_mesh(geometry, N) 

n_ref = 0
for it in range(n_ref):
    mesh_OT = refine(mesh_OT)
    mesh_c = refine(mesh_c)

N = mesh_OT.num_vertices()

coords = mesh_OT.coordinates()[:] 
ntasks = 10

step = coords.shape[0]//ntasks
idx_start_T = [idx*step for idx in range(ntasks)]


if coords.shape[0]%ntasks:
    idx_start_T += [step*ntasks,coords.shape[0]]
else:
    idx_start_T += [coords.shape[0]]


pool = mp.Pool() 
results = [pool.apply_async(solveOT, args = (idx_start_T[i],coords[idx_start_T[i]:idx_start_T[i+1],:],beta)) for i in range(len(idx_start_T)-1)]
pool_res = [p.get() for p in results]
pool.close()
pool.join()

for tup in pool_res:
    R_vec = tup[1]
    s_vec = tup[2]
    for i in range(len(R_vec)):
            idx = tup[0] + i
            R = R_vec[i]
            s = s_vec[i]
            
            x = coords[idx,0]
            y = coords[idx,1]
          
            if s == 0:    
                continue
            else:
                mesh_OT.coordinates()[idx,:] = (R/s)*np.array([x,y])
    
#print('total time passed: ',tot_time)
#string_mesh = 'mesh_OT/mesh_OT_' + str(N) + '_' + str(beta) + '.xml.gz'
#string_mesh = 'mesh_OT_crisscross/mesh_OT_' + str(N) + '.xml.gz'
#File(string_mesh) << mesh_OT
plot(mesh_OT)

X = FunctionSpace(mesh_c,'CG',1)
x_OT = Function(X)
y_OT = Function(X)

v_d = dof_to_vertex_map(X)

x_OT.vector()[:] = mesh_OT.coordinates()[v_d,0]
y_OT.vector()[:] = mesh_OT.coordinates()[v_d,1]

Q = skewness(mesh_c,x_OT,y_OT)

Q_scalar = np.max(Q.vector()[:])   
print(Q_scalar)


#Q = skewness(mesh_c,x_OT,y_OT)
#Q_scalar = np.max(Q.vector()[:])   
#print(Q_scalar)
#np.save('Data/OT/Q' + str(N) + '.npy',Q_scalar)

#File('Paraview/OT_mesh/mesh_OT'+ str(N) + '_' + str(beta) + '.pvd') << mesh_OT
#
#q = mesh_condition(mesh_OT)
#q_scalar = np.max(q.vector()[:])
#
#mu = shape_regularity(mesh_OT)
#mu_scalar = np.min(mu.vector()[:])

    
