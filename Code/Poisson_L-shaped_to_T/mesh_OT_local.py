#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:30:19 2021

1. Create locally the adapted meshes
2. Shift the local meshes to the proper sides
3. Take the boundary values and create a domain that is automatically meshed
    with the mshr package

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

tol = 1e-12

def Newton(gamma,s,x,eps,w=0.8):
    
    f_value = x**2 + x**(2*(1-gamma)) - s**2
    dfdx = 2*x + (2*(1-gamma))*x**(2*(1-gamma)-1)
    x_prev = x
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 500:
        
        try:
            x = w*x_prev  + (1-w)*(x - float(f_value)/dfdx)
            x_prev = x
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error
        
        f_value = x**2 + x**(2*(1-gamma)) - s**2
        dfdx = 2*x + (2*(1-gamma))*x**(2*(1-gamma)-1)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
        print('convergence not reached')
    
    return x, iteration_counter

output = 0
num = 1
# endpoint is included
#gamma_vec = np.linspace(0.0,0.5,num)
gamma_vec = np.array([0.5])

q_vec = np.zeros(num)
mu_vec = np.zeros(num) 

mesh = Mesh('ell_mesh.xml')
mesh.rotate(-90)
mesh.coordinates()[:] = mesh.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])
#mesh = refine(mesh)        
CG1 = FunctionSpace(mesh, "CG", 1) # function space for solution u

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
                
        #coeff = [A,B,gamma]
        R,it_counter = Newton(gamma,s,1e-8,eps=1e-12)
        mesh_OT.coordinates()[i,:] = np.array([R*x[0]/s,R*x[1]/s])
    
    
    domain_vertices = []
    hmax = mesh_OT.hmax()    
    
    V = FunctionSpace(mesh_OT, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())
    
    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]
     
    domain_vertices_1 = [] 
    domain_vertices_2 = [] 
    domain_vertices_3 = [] 
    domain_vertices_4 = [] 
    

    for x in mesh_OT.coordinates()[vertices_on_boundary,:]:
        
        # divide in four areas and order for increasing values of X
        if x[1] < -0.6:
            #print(vertex.point().array()[0], vertex.point().array()[1])
            domain_vertices_1.append([x[0], x[1]])
        
        elif x[0] < -0.6:
            #print(vertex.point().array()[0], vertex.point().array()[1])
            domain_vertices_2.append([x[0], x[1]])
        
        elif x[1] > 0.6:
            #print(vertex.point().array()[0], vertex.point().array()[1])
            domain_vertices_3.append([x[0], x[1]])
        
        elif x[0] > 0.6:
            #print(vertex.point().array()[0], vertex.point().array()[1])
            domain_vertices_4.append([x[0], x[1]])
      
        
    sorted_array_1 = np.array(domain_vertices_1).reshape(-1,2)
    #ind = np.argsort(sorted_array_1[:,0])  
    
    sorted_array_2 = np.array(domain_vertices_2).reshape(-1,2)
    #ind = np.argsort(sorted_array_2[:,1])  
    
    sorted_array_3 = np.array(domain_vertices_3).reshape(-1,2)
    ind = np.argsort(sorted_array_3[:,0])  
    sorted_array_3 = sorted_array_3[ind,:]
    
    sorted_array_4 = np.array(domain_vertices_4).reshape(-1,2)

    domain_vertices.append(Point(1.0, -1.0))
    
    for x in sorted_array_1:
        domain_vertices.append(Point(x[0]+1.0,x[1]))
    
    for x in sorted_array_2:
        domain_vertices.append(Point(x[0]+1.0,x[1]))    
    
    for x in sorted_array_3:
        domain_vertices.append(Point(x[0]+1.0,x[1]))
        
    for x in sorted_array_4:
        domain_vertices.append(Point(x[0]+1.0,x[1]))
    
    
    domain_vertices.append(Point(2.0,0.0))
    domain_vertices.append(Point(2.0,1.0))
    domain_vertices.append(Point(-2.0,1.0))
    domain_vertices.append(Point(-2.0,0.0))
    
    mesh_OT_2 = Mesh(mesh_OT)
    mesh_OT_2.coordinates()[:,0] = -mesh_OT_2.coordinates()[:,0] 
    
    V = FunctionSpace(mesh_OT_2, 'CG', 1)
    bc = DirichletBC(V, 1, DomainBoundary())
    u = Function(V)
    bc.apply(u.vector())
    
    # Get vertices sitting on boundary
    d2v = dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]

    domain_vertices_1 = [] 
    domain_vertices_2 = [] 
    domain_vertices_3 = [] 
    domain_vertices_4 = [] 
        
    for x in mesh_OT_2.coordinates()[vertices_on_boundary,:]:
            
            # divide in four areas and order for increasing values of X
            if x[0] < -0.6:
                #print(vertex.point().array()[0], vertex.point().array()[1])
                domain_vertices_1.append([x[0], x[1]])
            
            elif x[1] > 0.6:
                #print(vertex.point().array()[0], vertex.point().array()[1])
                domain_vertices_2.append([x[0], x[1]])
            
            elif x[0] > 0.6:
                #print(vertex.point().array()[0], vertex.point().array()[1])
                domain_vertices_3.append([x[0], x[1]])
            
            elif x[1] < -0.6:
                #print(vertex.point().array()[0], vertex.point().array()[1])
                domain_vertices_4.append([x[0], x[1]])
              
    
      
    sorted_array_1 = np.array(domain_vertices_1).reshape(-1,2)
    ind = np.argsort(sorted_array_1[:,1])
    sorted_array_1 = sorted_array_1[ind,:]
    
    sorted_array_2 = np.array(domain_vertices_2).reshape(-1,2)
    ind = np.argsort(sorted_array_2[:,0])  
    sorted_array_2 = sorted_array_2[ind,:]
    
    sorted_array_3 = np.array(domain_vertices_3).reshape(-1,2)[::-1]
    #ind = np.argsort(sorted_array_3[:,0])  
    #sorted_array_3 = sorted_array_3[ind,:]
    
    sorted_array_4 = np.array(domain_vertices_4).reshape(-1,2)[::-1]
     
    for x in sorted_array_1:
        domain_vertices.append(Point(x[0]-1.0,x[1]))
    
    for x in sorted_array_2:
        domain_vertices.append(Point(x[0]-1.0,x[1]))    
    
    for x in sorted_array_3:
        domain_vertices.append(Point(x[0]-1.0,x[1]))
        
    for x in sorted_array_4:
        domain_vertices.append(Point(x[0]-1.0,x[1]))
        
    domain_vertices.append(Point(-1.0, -1.0)) 

    
    N = 10
    # vertices listed in counter-clockwise order 
    geometry = mshr.Polygon(domain_vertices)
    mesh_3 = mshr.generate_mesh(geometry, N) 
    plot(mesh_3)
    
    ## Add manually new nodes and edges 
    
    mesh_OT.coordinates()[:,0] += 1.0
    mesh_OT_2.coordinates()[:,0] -= 1.0
    
    CG_OT = FunctionSpace(mesh_OT, 'CG', 1)
    CG_OT_2 = FunctionSpace(mesh_OT_2, 'CG', 1)
    CG_3 = FunctionSpace(mesh_3, 'CG', 1)
    
    dofmap_OT = CG_OT.dofmap()
    coords_OT = CG_OT.tabulate_dof_coordinates()
    
    dofmap_OT_2 = CG_OT_2.dofmap()
    coords_OT_2 = CG_OT_2.tabulate_dof_coordinates()
    
    dofmap_3 = CG_3.dofmap()
    coords_3 = CG_3.tabulate_dof_coordinates()
    
    nvertices_OT = mesh_OT.num_vertices() 
    nvertices_OT_2 = mesh_OT_2.num_vertices() 
    nvertices_3 = mesh_3.num_vertices() 
    
    ncells_OT = mesh_OT.num_cells() 
    ncells_OT_2 = mesh_OT_2.num_cells()
    ncells_3 = mesh_3.num_cells()
    
    editor = MeshEditor()
    mesh_T = Mesh()
    editor.open(mesh_T,'triangle', 2, 2)  # top. and geom. dimension are both 2
    editor.init_vertices(nvertices_OT + nvertices_OT_2 + nvertices_3)  # number of vertices
    editor.init_cells(ncells_OT + ncells_OT_2 + ncells_3)     # number of cells
    
    tol = 1e-12
    
    
    for i in range(nvertices_OT):
        
        editor.add_vertex(i,np.array([coords_OT[i,0],coords_OT[i,1]]))
        
    
    for i in range(nvertices_OT_2):
        
        editor.add_vertex(i + nvertices_OT,np.array([coords_OT_2[i,0], coords_OT_2[i,1]]))
    
    
    for i in range(nvertices_3):
        
        editor.add_vertex(i + nvertices_OT + nvertices_OT_2,np.array([coords_3[i,0], coords_3[i,1]]))
    
        
    for i in range(ncells_OT):
        
        editor.add_cell(i,dofmap_OT.cell_dofs(i))
    
    for i in range(ncells_OT_2):
        
        editor.add_cell(ncells_OT + i, dofmap_OT_2.cell_dofs(i) + nvertices_OT)
    
    
    for i in range(ncells_3):
        
        editor.add_cell(ncells_OT + ncells_OT_2 + i, dofmap_3.cell_dofs(i) +  nvertices_OT + nvertices_OT_2)
    
    editor.close(order=True)
        
    f = File('mesh.pvd')
    f << mesh
    
    plot(mesh_T)
    q = mesh_condition(mesh_T)
    mu = shape_regularity(mesh_T)
   
    q_vec[it] = np.max(q.vector()[:])
    mu_vec[it] = np.min(mu.vector()[:])   
     
#    
#    string_mesh = 'Mesh/mesh_OT_' + str(np.round(gamma, 2)) + '.xml.gz'
#    File(string_mesh) << mesh_OT
