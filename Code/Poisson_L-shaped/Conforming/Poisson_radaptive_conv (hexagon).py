#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

Solve Poisson equation in region with corner singularity 

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
    
# Expression for exact solution 
class Expression_grad_uexact(UserExpression):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs) # This part is new!
    
    def eval(self, value, x):
        
        r = sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
        
        if r == 0.0:    
            value[0] = 0.0
            value[1] = 0.0
        else:
            
            value[0] = -(2.0/3.0)*pow(r,-1.0/3.0)*sin(1.0/3.0*theta)
            value[1] = (2.0/3.0)*pow(r,-1.0/3.0)*cos(1.0/3.0*theta)
            
    def value_shape(self):
        return (2,)    


def solve_poisson(u_expr):
    
    n = FacetNormal(mesh)
    
    v = TestFunction(CG1)
    u = TrialFunction(CG1)
    
    bcs = DirichletBC(CG1, u_expr,'on_boundary')
    
    # Weak form 
    a = inner(grad(v),grad(u))*dx(domain = mesh)
    L = Constant(0.0)*v*dx(domain = mesh)
    
    u = Function(CG1)
    solve(a==L,u,bcs)
    
    return u



def monitor():
    
    dx = Measure('dx',domain = mesh)
    dS = Measure('dS',domain = mesh)
    area = assemble(Constant(1.0)*dx(mesh))
    
    if monitor_str == 'a-posteriori':
        
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        w = TestFunction(DG0)
        
        residual = Function(DG0)
      
        # assume avg diameter close to edge length
        residual_tensor = h**2*w*(div(grad(u)))**2*dx + avg(w)*avg(h)*jump(grad(u),n)**2*dS
        assemble(residual_tensor, tensor=residual.vector())
        residual = interpolate(residual,CG1)
        
        alpha =  np.power((1/area)*assemble(pow(residual,0.5)*dx(mesh)),2.0)
      
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + (1/alpha)*(residual.vector()[:]))    
        
    elif monitor_str == 'curvature':
        
        grad_u = project(grad(u),FunctionSpace(mesh,'RT',1)) 
        laplacian = project(div(grad_u),CG1) 
       
        #alpha = pow((1/area)*assemble(pow(abs(laplacian),0.5)*dx(mesh)),2)
        
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + (np.abs(laplacian.vector()[dof_P_to_C])))   
        
    elif monitor_str == 'gradient':
        
        # Project solution gradient in CG1 space 
        grad_u = project(grad(u), VectorFunctionSpace(mesh, 'CG', 1))
        
        ux = project(grad(u)[0],CG1)
        uy = project(grad(u)[1],CG1)
        
        alpha = np.power((1/area)*assemble(pow(dot(grad_u,grad_u),0.5)*dx(mesh)),2) 
        
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + 1000*(ux.vector()[dof_P_to_C]**2 + uy.vector()[dof_P_to_C]**2))
    
    return w

# smoothing gives smoother increment of the local skewness
def smoothing(w,beta):
    
    w_test = TestFunction(X)
    w_trial = TrialFunction(X) 
    
    dx = Measure('dx',domain = mesh_c)
    a = w_test*w_trial*dx + beta*inner(grad(w_test),grad(w_trial))*dx 
    L = w_test*w*dx

    w = Function(X)
    solve(a==L,w)
    
    # scale
    #w.vector()[:] = w.vector()[:]/np.max(w.vector()[:])

    return w


def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/ln(sqrt(dof[i+1]/dof[i]))

    return rate



def equidistribute(x,rho,tau,dt,dxi):
    
    '''
        input parameters:
            x: coordinates in physical domain 
            w: monitor function
            
        output parameters:
            y: equidistributed coordinates 
            
        Solve MMPDE5 using finite difference along the boundary
        Solve linear system Ax = b where
        
        A = [ -alpha r-  (1 + alpha r+  + alpha r-)   -alpha r+ ] x^n+1 = x^ n
        
        A[0,:] = [1 0 ... 0]
        A[n,:] = [0 ... 0 1]
        
    '''

    # number of mesh points counted from 0 ton nx-1
    nx = x.shape[0]
    
    v = x.copy()
    v = np.array(sorted(v))
    
    dof_to_sort = [np.where(x == v[i])[0].tolist() for i in range(nx)]
    dof_to_sort = np.hstack(dof_to_sort)
    
    sort_to_dof = [np.where(v == x[i])[0].tolist() for i in range(nx)]
    sort_to_dof = np.hstack(sort_to_dof)
    
    rho = rho[dof_to_sort]   
    
    A = np.zeros((nx,nx))   
    alpha = np.zeros(nx)
    rp =  np.zeros(nx)
    rm =  np.zeros(nx)
    
    alpha[1:-1] = dt/(rho[1:-1]*tau*(dxi)**2)
    
    rp[1:-1] = (rho[1:-1] + rho[2:])/2.0
    rm[1:-1] = (rho[:-2] + rho[1:-1])/2.0
    
    A[0,0] = A[nx-1,nx-1] = 1.0 
    
    for i in range(1,nx-1):
        A[i,(i-1):(i+2)] = [- alpha[i]*rm[i], 1 + alpha[i]*rp[i] + alpha[i]*rm[i],-alpha[i]*rp[i]] 
    
    
    v_new = np.linalg.solve(A,v)
    
    v_new[0] = 0.0
    v_new[nx-1] = 1.0

    return v_new[sort_to_dof]



def boundary_mesh(x_old,y_old,w,tau,dt,dxi):
   
    y1 = y_old.vector()[dofs_1]
    w1 = w.vector()[dofs_1]
    
    y1 = equidistribute(y1,w1,tau,dt,dxi)
    
    Y_boundary.vector()[dofs_1] = y1
    
    x2 = x_old.vector()[dofs_2]
    w2 = w.vector()[dofs_2]
    
    x2 = equidistribute(x2,w2,tau,dt,dxi)
    X_boundary.vector()[dofs_2] = x2

    return X_boundary,Y_boundary



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


## Define domain boundaries
def boundary_1c(x, on_boundary):
     return on_boundary and near(x[1] + sqrt(3)*x[0] + sqrt(3), 0.0, tol)

def boundary_2c(x, on_boundary):
     return on_boundary and near(x[1], -sqrt(3)/2, tol)

def boundary_3c(x, on_boundary):
     return on_boundary and near(x[1] - sqrt(3)*x[0] + sqrt(3),0.0, tol)

def boundary_4c(x, on_boundary):
     return on_boundary and near(x[1] + sqrt(3)*x[0] - sqrt(3),0.0, tol)

def boundary_5c(x, on_boundary):
     return on_boundary and near(x[1], sqrt(3)/2, tol)

def boundary_6c(x, on_boundary):
     return on_boundary and near(x[1] - sqrt(3)*x[0] - sqrt(3),0.0, tol)




monitor_str = 'curvature'
parameters['allow_extrapolation'] = True


Ns = np.array([2**4])
tol = 1e-16
dt = 1e-2
tau = 1.0
itmax = 150
beta = 1e-4



## Create L-shaped domain 
rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2

output = 1
Energy_norm = np.zeros(Ns.shape[0])
dof = np.zeros(Ns.shape[0])    

for i,N in enumerate(Ns):
    
    dxi = 1/N
    mesh = mshr.generate_mesh(geometry, Ns)   # physical mesh 
 
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
 
    u_expr = Expression_uexact(degree=5)    
    gradu_expr = Expression_grad_uexact(degree=5)  
    
    u = solve_poisson(u_expr)
    
    ####  Solve Laplace  ######
    
    xi = Function(CG1)
    nu = Function(CG1)
    
    xi_trial = TrialFunction(CG1)
    nu_trial = TrialFunction(CG1)
    
    xi_test = TestFunction(CG1)
    nu_test = TestFunction(CG1)
    
    # impose bc for xi,nu
    
    bc_xi = [DirichletBC(CG1,Expression('0.5*x[0]-0.5',degree=1),boundary_1),DirichletBC(CG1,Expression('x[1]+0.5',degree=1),boundary_2),DirichletBC(CG1,Expression('0.5*x[0]+0.5',degree=1),boundary_3),
             DirichletBC(CG1,Expression('-0.5*x[1]+1',degree=1),boundary_4),DirichletBC(CG1,Expression('0.5*x[0]',degree=1),boundary_5),DirichletBC(CG1,Expression('0.25*x[1]-0.75',degree=1),boundary_6)]
    
    
    bc_nu = [DirichletBC(CG1,Expression('-sqrt(3)/2*(x[0]+1)',degree=1),boundary_1),DirichletBC(CG1,-sqrt(3)/2,boundary_2),DirichletBC(CG1,Expression('sqrt(3)/2*(x[0]-1)',degree=1),boundary_3),
             DirichletBC(CG1,Expression('sqrt(3)/2*x[1]',degree=1),boundary_4),DirichletBC(CG1,sqrt(3)/2,boundary_5),DirichletBC(CG1,Expression('sqrt(3)/4*(x[1]+1)',degree=1),boundary_6)]
    
    
    a = inner(grad(xi_test),grad(xi_trial))*dx
    L = Constant(0.0)*xi_test*dx
    
    solve(a==L,xi,bc_xi)
    
    a = inner(grad(nu_test),grad(nu_trial))*dx
    L = Constant(0.0)*nu_test*dx
    
    solve(a==L,nu,bc_nu)
    
     
    file_xi = File('Paraview_p2/xi_' + monitor_str + '.pvd')
    file_nu = File('Paraview_p2/nu_'+ monitor_str + '.pvd')

    file_xi << xi
    file_nu << nu 
    
    coords_P = CG1.tabulate_dof_coordinates()
    dofmap_P = CG1.dofmap()

    nvertices = mesh.num_vertices()
    ncells = mesh.num_cells()
    
    editor = MeshEditor()
    mesh_c = Mesh()
    editor.open(mesh_c,'triangle', 2, 2)  # top. and geom. dimension are both 2
    editor.init_vertices(nvertices)  # number of vertices
    editor.init_cells(ncells)     # number of cells
    
    for i in range(nvertices):
        editor.add_vertex(i,np.array([xi(coords_P[i,:]),nu(coords_P[i,:])]))
    
    for i in range(ncells):
        editor.add_cell(i,dofmap_P.cell_dofs(i))
        #dtype=np.uintp
    editor.close()

    X = FunctionSpace(mesh_c,'CG',1) 
    DG0_c = FunctionSpace(mesh_c,'DG',0) 
    
    coords_C = X.tabulate_dof_coordinates()
    dofmap_C = X.dofmap()
    
    ## Smooth monitor function 
    #w = smoothing(w_old,beta)
    
    ## get map from Physical to Computational domain and viceversa
    A = np.array([[xi(coords_P[i,:]),nu(coords_P[i,:])] for i in range(nvertices)])
    
    dof_C_to_P = np.zeros(nvertices)
    dof_P_to_C = np.zeros(nvertices)
    
    for i in range(nvertices):
        for j in range(nvertices):
            if (coords_C[i,0] == A[j,0]) and (coords_C[i,1]== A[j,1]):
               dof_C_to_P[j] = i 
               dof_P_to_C[i] = j
                
    ## monitor function 
    w = monitor()
    
    ## Solve MMPDE based on Winslow's equation
    
    x_trial = TrialFunction(X)
    y_trial = TrialFunction(X)
    
    x_test = TestFunction(X)
    y_test = TestFunction(X)
    
    x_old = Function(X)
    y_old = Function(X)
    uc = Function(X)
    
    xp = interpolate(Expression('x[0]',degree=1),CG1)
    yp = interpolate(Expression('x[1]',degree=1),CG1)
    
    x_old.vector()[:] = xp.vector()[dof_P_to_C]
    y_old.vector()[:] = yp.vector()[dof_P_to_C]
    uc.vector()[:] = u.vector()[dof_P_to_C]    
    
    x_b = x_old.copy(deepcopy = True)
    y_b = y_old.copy(deepcopy = True)
    
    X_boundary = Function(X)
    Y_boundary = Function(X)
         
    file_xc = File('Paraview_p2/xc_' + monitor_str + '.pvd')
    file_yc = File('Paraview_p2/yc_'+ monitor_str + '.pvd')
    file_uc = File('Paraview_p2/uc_'+ monitor_str + '.pvd')

    file_xc << x_old
    file_yc << y_old
    file_uc << uc


    x_new = Function(X)
    y_new = Function(X)
    
    dofs_1 = np.intersect1d(np.where(x_old.vector() == 0.0)[0],np.where(y_old.vector() <= 0.0)[0])    
    dofs_2 = np.intersect1d(np.where(y_old.vector() == 0.0)[0],np.where(x_old.vector() >= 0.0)[0])

    x0 = Function(X)
    x1 = Function(X)
    
    y0 = Function(X)
    y1 = Function(X)
    
    Ax = assemble(x_trial*x_test*dx(domain = mesh_c))
    Ay = assemble(y_trial*y_test*dx(domain = mesh_c))
    
    if output:
        
        file_u = File('Paraview_p2/u_' + monitor_str + '.pvd')
        file_w = File('Paraview_p2/w_'+ monitor_str + '.pvd')
        
        u.rename('u','u')
        w.rename('w','w')
        
        file_u << u 
        file_w << w
   
    it = 0
    err = np.zeros(itmax)
    
    X_boundary = x_old.copy()
    Y_boundary = y_old.copy()
        
    while (it < itmax):
        
      print('it: ',it)
      X_boundary,Y_boundary = boundary_mesh(x_old,y_old,w,tau,dt,dxi)
      bc_x = [DirichletBC(X,X_boundary,'on_boundary'),DirichletBC(X,0.0,boundary_2c),DirichletBC(X,1.0,boundary_4c),DirichletBC(X,-1.0,boundary_6c)]
      bc_y = [DirichletBC(X,Y_boundary,'on_boundary'),DirichletBC(X,-1.0,boundary_1c),DirichletBC(X,0.0,boundary_3c),DirichletBC(X,1.0,boundary_5c)]
      #bc_x = DirichletBC(X,x_b,'on_boundary')
      #bc_y = DirichletBC(X,y_b,'on_boundary')
        
      
      dx = Measure('dx',domain=mesh_c)
        
      # Project derivatives of x,y in CG space
      Lx0 = assemble(x_old.dx(0)*x_test*dx)
      Lx1 = assemble(x_old.dx(1)*x_test*dx)
      Ly0 = assemble(y_old.dx(0)*y_test*dx)
      Ly1 = assemble(y_old.dx(1)*y_test*dx)
    
      solve(Ax,x0.vector(),Lx0)  
      solve(Ax,x1.vector(),Lx1)  
      
      solve(Ay,y0.vector(),Ly0)  
      solve(Ay,y1.vector(),Ly1)  
          
      alpha_1 = x1*x1 + y1*y1
      alpha_2 = x0*x1 + y0*y1
      alpha_3 = x0*x0 + y0*y0
      
      J = x0*y1 - x1*y0
      
      p = sqrt(2)*sqrt(alpha_1+alpha_1 + 2*alpha_2*alpha_2 + alpha_3*alpha_3)/(J*J*w)
      A = (J*J)*w*w*tau*p
      
      # Lagged backward Euler scheme 
      
      #(K1 - dtK2)C^n+1 = K1*C^n
      
      K1 = A*x_trial*x_test*dx
      K2 = w*(-(alpha_1*x_test).dx(0)*x_trial.dx(0) + ((alpha_2*x_test).dx(1)*x_trial.dx(0) +
                                                       (alpha_2*x_test).dx(0)*x_trial.dx(1)) -
              (alpha_3*x_test).dx(1)*x_trial.dx(1))*dx
      
      a = K1 - dt*K2
      L = A*x_old*x_test*dx
      
      solve(a==L,x_new,bc_x,solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})
      
      
      K1 = A*y_trial*y_test*dx
      K2 = w*(-(alpha_1*y_test).dx(0)*y_trial.dx(0) + ((alpha_2*y_test).dx(1)*y_trial.dx(0) + 
                                                       (alpha_2*y_test).dx(0)*y_trial.dx(1)) - 
              (alpha_3*y_test).dx(1)*y_trial.dx(1))*dx
      
      a = K1 - dt*K2
      L = A*y_old*y_test*dx
      
      solve(a==L,y_new,bc_y,solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})
      
      x_old = x_new
      y_old = y_new
      
      xp.vector()[:] = x_new.vector()[dof_C_to_P]
      yp.vector()[:] = y_new.vector()[dof_C_to_P]
      
      # update mesh coordinates 
      mesh.coordinates()[:,0] = xp.compute_vertex_values()
      mesh.coordinates()[:,1] = yp.compute_vertex_values()
      
      u = solve_poisson(u_expr)
      
      w = monitor()
      #w = smoothing(w,beta)
      
      if output:
          u.rename('u','u')
          w.rename('w','w')
          
          file_u << u
          file_w << w
      
      err[it] = np.sqrt(assemble(dot(u_expr - u,u_expr - u)*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 

      it += 1      
        

# For CG1 convergence rate in H1 semi-norm is expected to be 1 
#rate = conv_rate(dof,Energy_norm)
fig, ax = plt.subplots()
ax.plot(err,linestyle = '-.',marker = 'o',label = 'r-adaptive - gradient')
#ax.plot(dof_movmesh_aposteriori,H10_movmesh_aposteriori,linestyle = '-.',marker = 'o',label = 'blah')
ax.set_xlabel('iterations')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')           
     

np.save('H10_movmesh_' + monitor_str + '.npy',Energy_norm)
np.save('dof_movmesh_' + monitor_str + '.npy',dof)
np.save('rate_movmesh_' + monitor_str + '.npy',rate)
