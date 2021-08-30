#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:50:33 2021

@author: simone

Solve of the OT mesh for the L-shaped domain 
Shift the coordinates and elimate the nodes at the location x == 0
Obtain the adapted T-shaped domain 


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
num = 3
# endpoint is included
gamma_vec = np.linspace(0.0,0.5,num)
# fix refinement for different gammas
n_ref = 1

q_vec = np.zeros(num)
mu_vec = np.zeros(num) 

mesh = Mesh('ell_mesh.xml')
mesh.rotate(-90)
mesh.coordinates()[:] = mesh.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])
#plot(mesh_uniform)

for j in range(n_ref):
 
    print(' Refinement n° ',j+1)
  
  # first refine uniform mesh and then apply OT mesh 
    mesh = refine(mesh)        
    V = FunctionSpace(mesh, "DG", 1) # function space for solution u
    dof = V.dim()   
      
coords_uniform = mesh.coordinates()[:]
nVertices = coords_uniform.shape[0]

for it,gamma in enumerate(gamma_vec): 
    
    print('Iteration for gamma: ', gamma)    
        
    # create deep copy of uniform mesh
    mesh_OT = Mesh(mesh)
      
    if output:
        File_q = File('Paraview/OT_priori/q_'+ str(np.round(gamma, 2)) + '.pvd')
        File_mu = File('Paraview/OT_priori/mu_'+ str(np.round(gamma, 2)) + '.pvd')

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
    
    q = mesh_condition(mesh_OT)
    mu = shape_regularity(mesh_OT)
   
    q_vec[it] = np.max(q.vector()[:])
    mu_vec[it] = np.min(mu.vector()[:])   
 
    
    string_mesh = 'Mesh/mesh_OT_' + str(np.round(gamma, 2)) + '.xml.gz'
    File(string_mesh) << mesh_OT

mesh_OT.coordinates()[:,0] = mesh_OT.coordinates()[:,0] +1.0
        
    
# Once you obtained the adapted mesh create a copy and flip it 
mesh_OT_2 = Mesh(mesh_OT)
mesh_OT_2.coordinates()[:,0] = - mesh_OT_2.coordinates()[:,0]
#
#plt.figure()
#plot(mesh_OT)
#
#plt.figure()
#plot(mesh2)


CG_OT = FunctionSpace(mesh_OT, 'CG', 1)
CG_OT2 = FunctionSpace(mesh_OT_2, 'CG', 1)

dofmap_OT = CG_OT.dofmap()
coords_OT = CG_OT.tabulate_dof_coordinates()

dofmap_OT_2 = CG_OT2.dofmap()
coords_OT_2 = CG_OT2.tabulate_dof_coordinates()

# Join the 2 meshes 
nvertices_OT = mesh_OT.num_vertices() 
nvertices_OT_2 = mesh_OT_2.num_vertices() 

ncells_OT = mesh_OT.num_cells() 
ncells_OT_2 = mesh_OT_2.num_cells()

editor = MeshEditor()
mesh_T = Mesh()
editor.open(mesh_T,'triangle', 2, 2)  # top. and geom. dimension are both 2
editor.init_vertices(nvertices_OT + nvertices_OT_2)  # number of vertices
editor.init_cells(ncells_OT + ncells_OT_2)     # number of cells

tol = 1e-12


for i in range(nvertices_OT):
    
    editor.add_vertex(i,np.array([coords_OT[i,0],coords_OT[i,1]]))
    

for i in range(nvertices_OT_2):
    
    editor.add_vertex(i + nvertices_OT,np.array([coords_OT_2[i,0], coords_OT_2[i,1]]))

    
for i in range(ncells_OT):
    
    editor.add_cell(i,dofmap_OT.cell_dofs(i))

for i in range(ncells_OT_2):
    
    editor.add_cell(ncells_OT + i, dofmap_OT_2.cell_dofs(i) + nvertices_OT)

    
editor.close()

plt.figure()
plot(mesh_T)

if output:
    File_q << q,it
    File_mu << mu,it
   
np.save('Data/q.npy',q_vec)
np.save('Data/mu.npy',mu_vec)


dict = {'gamma': gamma_vec, 'mu': mu_vec, 'q': q_vec}
df = pd.DataFrame(dict) 
df.to_csv('Data/stat.csv',index=False) 
