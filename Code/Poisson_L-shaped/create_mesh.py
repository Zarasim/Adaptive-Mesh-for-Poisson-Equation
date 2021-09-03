#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:12:31 2021

@author: simone
"""

from dolfin import *
import numpy as np


#  684  1716  7005  28290  100080  

# 22 771 1452
# 20 641 1200

N = 20

mesh = RectangleMesh(Point(-1.0,-1.0),Point(1.0, 1.0), N, N, "crossed")
CG1 = FunctionSpace(mesh, 'CG', 1)

coords_P = CG1.tabulate_dof_coordinates()
dofmap_P = CG1.dofmap()

nvertices = mesh.num_vertices()
ncells = mesh.num_cells()


editor = MeshEditor()
mesh_L = Mesh()
editor.open(mesh_L,'triangle', 2, 2)  # top. and geom. dimension are both 2
editor.init_vertices(641)  # number of vertices
editor.init_cells(1200)     # number of cells

B_L = []

counter_v = 0
for i in range(nvertices):
    if (coords_P[i,0] > 0.0) and (coords_P[i,1] < 0.0):
        continue
    
    B_L.append(i)
    editor.add_vertex(counter_v,coords_P[i,:])
    counter_v += 1

B_L = np.array(B_L)    

counter = 0
for i in range(ncells):
    con = dofmap_P.cell_dofs(i)
    if np.any(coords_P[con,0] > 0.0) and np.any(coords_P[con,1] < 0.0):
        continue
    
    conn = [int(np.where(B_L == v)[0]) for v in con]
    editor.add_cell(counter,conn)
    counter += 1
    
editor.close()


string_mesh = 'mesh_uniform/mesh_uniform_' + str(counter_v) + '.xml.gz'
File(string_mesh) << mesh_L
plot(mesh_L)