#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:56:11 2021

@author: simone

Define mesh in L-shaped domain with radial simmetry

1) Fix a parameter delta theta to define a line starting from the origin  
2) For each line find boundary point and solve equidistribution eq. in 1D  
3) For sides intersecting in reentrant corner equidistribute separately
4) Construct the mesh and store it in a xml file

"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import mshr 
import math

# Expression for exact solution 
class Expression_u(UserExpression):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs) # This part is new!
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        if r == 0.0:
            value[0] = 0.0
        else:
            value[0] = pow(r,2.0/3.0)*sin(2.0*theta/3.0) 
    def value_shape(self):
        return ()
        
    
class Expression_u_1D(UserExpression):
    
    def __init__(self,theta,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.theta = theta
    
    # r is defined through x[1] = tg(theta)*x[0]
    def eval(self, value, x):
        
       
        value[0] = pow(x[0],2.0/3.0)*sin(2.0*self.theta/3.0) 
    
    def value_shape(self):
        return ()
        
    
def length_line(theta):
    
    '''
    
    Define length of the line that is multiplied to UnitIntervalMesh
    
    Exclude Boundary 2 and 3
    '''
    # x[0] = 1
    if (theta >= 0) and (theta <= pi/4):
        return sqrt(1 + math.tan(theta)**2)
    # x[1] = 1
    elif (theta > pi/4) and (theta <= 3.0/4.0*pi):
        return sqrt(1 + (1/math.tan(theta))**2)
    # x[0] = -1
    elif (theta > 3.0/4.0*pi) and (theta <= 5.0/4.0*pi):
        return sqrt(1 + abs(math.tan(theta))**2)
    # x[1] = -1 
    elif (theta > 5.0/4.0*pi) and (theta <= 3.0/2.0*pi):
        return sqrt(1 + (1/abs(math.tan(theta)))**2)
    

# Define domain boundaries
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


def monitor(mesh,u_1D,length_side):
      
    # Optimal mesh density function in 1D
    w = Function(V)
    u1d_proj = project(u_1D.dx(0).dx(0),V)
    alpha = (assemble(np.power(u1d_proj,2.0/5.0)*dx)/length_side)**5
    w.vector()[:] = np.power(1 + (1/alpha)*(u1d_proj.vector()[:])**2,1.0/5.0)

    return w

def monitor_interpolant(mesh,u_ex):
      
    diam = CellDiameter(mesh)
    x = mesh.coordinates()[:,0]
    w = TestFunction(DG0)
    avg_u_squared = Function(DG0) 
    
    uxx = project(u_ex.dx(0).dx(0),V)
   
    # assume avg diameter close to edge length
    u_form = w*(1/diam)*(uxx)*(uxx)*dx(mesh)
    assemble(u_form, tensor=avg_u_squared.vector())
    
    #alpha = 1.0
    alpha_h = np.sum(np.diff(x)*np.power(avg_u_squared.vector()[:],1.0/5.0))**5   
    avg_u_squared = interpolate(avg_u_squared,V)
    
    # w must be interpolated in the computational mesh for integration
    w_interp = Function(V)
    w_interp.vector()[:] = np.power((1 + (1/alpha_h)*avg_u_squared.vector()[:]),1.0/5.0)

    return w_interp


def equidistribute(x,w,dof2vertex_map):
    
    rho = w.vector()[dof2vertex_map]
    # Make a copy of vector x to avoid overwriting
    y = x.copy()
        
    # number of mesh points counted from 0 ton nx-1
    nx = x.shape[0]
    

    II = nx - 1 
    JJ = nx - 1
    
    # Create vector of integrals with nx entries 
    intMi = np.zeros(nx)
    
    
    # compute each integral using trapezoidal rule
    intMi[1:] = 0.5*(rho[1:] + rho[:-1])*np.diff(x)
    
    # take cumulative sum of integrals
    intM = np.cumsum(intMi)
    
    # take total integral theta
    theta = intM[-1]
    
    
    jj = 0
    
    # Assign new nodes from  y_1 to y_(nx - 2)
    for ii in range(1,II):
        
        # Target =  y_1 = 1/(nx-1)*theta ... y_nx-2 = (nx-2)/(nx-1)*theta
    
        Target = ii/II*theta
        
    
        while jj < JJ and intM[jj] < Target:
        
            jj = jj+1
            
        jj = jj - 1
        
        Xl = x[jj]
        Xr = x[jj+1]
        Ml = rho[jj]
        Mr = rho[jj+1]
        
        Target_loc = Target - intM[jj]
        
        mx = (Mr - Ml)/(Xr - Xl)
        
        y[ii] = Xl + 2*Target_loc/(Ml + np.sqrt(Ml**2 + 2*mx*Target_loc))
        
        y[0] = np.min(x)
        y[-1] = np.max(x)
        
    return y



def quality_measure(mesh):
    
    ## Compute mesh Skewness
    mu = Function(DG0)
    
    for c in cells(mesh):       
        
        pk = c.inradius()
        hk = c.h()
        
        mu.vector()[c.index()] = pk/hk
    
    return mu



# define dtheta
N = 70
theta_num = 70
theta_vec,theta_step = np.linspace(0.0,(3.0/2.0)*pi,retstep=True,num=theta_num)

mesh_nodes = np.zeros([theta_num,N])

reltol = 1e-12
max_iter = 30
for i,theta in enumerate(theta_vec):
        
     if i == 0:
         theta = 1e-3
     elif i == (theta_num-1):
         theta = (3.0/2.0)*pi - 1e-3
    
     length_side = length_line(theta)
     mesh = UnitIntervalMesh(N-1)
     u_expr_1D = Expression_u_1D(theta,degree=5)
     
     V = FunctionSpace(mesh,'CG',1) 
     CG5 = FunctionSpace(mesh,'CG',5) 
     DG0 = FunctionSpace(mesh,'DG',0) 
     dof2vertex_map = dof_to_vertex_map(V)
     
     tol = 1.0
     iteration = 0
     
     u_1D = interpolate(u_expr_1D,CG5)
     
     x = mesh.coordinates()[:,0]*length_side
    
     while (tol > reltol) and (iteration < max_iter):
        
        print('rel_tol is: ',tol)
        
        w_post = monitor_interpolant(mesh,u_1D)
   
        x_new = equidistribute(x,w_post,dof2vertex_map)
        
        tol = max(abs(x_new - x))/max(x)
        
        mesh.coordinates()[:,0] = x_new
        u_1D = interpolate(u_expr_1D,CG5)
        
        x = x_new
        
        iteration +=1
        
    # collect mesh nodes in a matrix 
     mesh_nodes[i,:] = x

# update first and last row
#mesh = UnitIntervalMesh(N-1)
mesh_nodes[0,-1] = 1.0
mesh_nodes[-1,-1] = 1.0


# Construct mesh by adding vertices and cells

editor = MeshEditor()
mesh_c = Mesh()
editor.open(mesh_c,'triangle', 2, 2)  # top. and geom. dimension are both 2

nvertices = np.size(mesh_nodes[:,1:])+4

editor.init_vertices(nvertices)  # number of vertices
editor.init_cells(9456)     # number of cells

# 12x12 -> 234
# 20 x 20 -> 706
# 40 x 40 -> 3006
# 70x70 -> 9456
# 150x150 -> 44256

editor.add_vertex(0,np.array([0.0,0.0]))

## add vertex (1.0,1.0),(-1.0,1.0),(-1.0,-1.0)

vertex_1 = 0
vertex_2 = 0
vertex_3 = 0

counter = 1

for i in range(1,N):
    for j in range(theta_num):
        
        x = mesh_nodes[j,i]
        # x[0] = x*cos(theta)  x[1] = x*sin(theta)
        if (abs(theta_vec[j]-pi/4) < theta_step) and (vertex_1 < 1) and (i==N-1): 
            editor.add_vertex(counter,np.array([x*math.cos(theta_vec[j]),x*math.sin(theta_vec[j])]))
            idx_tr1 = counter
            counter += 1
            vertex_1 += 1
            
        elif (abs(theta_vec[j]-3.0/4.0*pi) < theta_step) and (vertex_2 < 1) and (i==N-1): 
            
            editor.add_vertex(counter,np.array([x*math.cos(theta_vec[j]),x*math.sin(theta_vec[j])]))
            idx_tl1 = counter
            counter += 1
            vertex_2 += 1
        
        elif (abs(theta_vec[j]-5.0/4.0*pi) < theta_step) and (vertex_3 < 1) and (i==N-1): 
            
            editor.add_vertex(counter,np.array([x*math.cos(theta_vec[j]),x*math.sin(theta_vec[j])]))
            idx_bl1 = counter
            counter += 1
            vertex_3 += 1
        else:
            editor.add_vertex(counter,np.array([x*math.cos(theta_vec[j]),x*math.sin(theta_vec[j])]))
            counter += 1

idx_tr = counter
editor.add_vertex(counter,np.array([1.0,1.0]))
counter += 1

idx_tl = counter
editor.add_vertex(counter,np.array([-1.0,1.0]))
counter += 1

idx_bl = counter
editor.add_vertex(counter,np.array([-1.0,-1.0]))

# first cells around centre
for i in range(1,theta_num):
    cell_dofs = np.array([0,i,i+1])
    editor.add_cell(i-1,cell_dofs)

# define other cells 
counter = theta_num-1

for i in range(1,N-1):
    for j in range(theta_num-1):
        # define indeces of quadrilateral
 
        j1 = 1 + (i-1)*theta_num +j
        j2 = j1+1
        
        j3 = j1 + theta_num
        j4 = j2 + theta_num
        
        cell_dofs = np.array([j1,j3,j4])
        editor.add_cell(counter,cell_dofs)
        counter += 1
        
        cell_dofs = np.array([j1,j2,j4])
        editor.add_cell(counter,cell_dofs)
        counter += 1

cell_dofs = np.array([idx_tr1,idx_tr1+1,idx_tr])
editor.add_cell(counter,cell_dofs)
counter += 1

cell_dofs = np.array([idx_tl1,idx_tl1+1,idx_tl])
editor.add_cell(counter,cell_dofs)
counter += 1

cell_dofs = np.array([idx_bl1,idx_bl1+1,idx_bl])
editor.add_cell(counter,cell_dofs)


editor.close()

DG0 = FunctionSpace(mesh_c,'DG',0)
DG1 = FunctionSpace(mesh_c,'DG',1)
mu = quality_measure(mesh_c)
File_mesh = File('Paraview/mesh_radial/mesh' + str(nvertices) + '.pvd')     

string_mesh = 'my_mesh' + str(nvertices) + '.xml.gz'
File(string_mesh) << mesh_c
File_mesh << mu
# 

# import mesh 
#mesh = Mesh(string_mesh)



