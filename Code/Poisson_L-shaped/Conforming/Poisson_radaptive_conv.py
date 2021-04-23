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
       
        alpha = pow((1/area)*assemble(pow(abs(laplacian),0.5)*dx(mesh)),2)
        
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + (1.0/alpha)*(np.abs(laplacian.vector()[:])))   
        
    elif monitor_str == 'gradient':
        
        # Project solution gradient in CG1 space 
        grad_u = project(grad(u), VectorFunctionSpace(mesh, 'CG', 1))
        
        ux = project(grad(u)[0],CG1)
        uy = project(grad(u)[1],CG1)
        
        alpha = np.power((1/area)*assemble(pow(dot(grad_u,grad_u),0.5)*dx(mesh)),2) 
        
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + (1/alpha)*(ux.vector()[:]*ux.vector()[:] + uy.vector()[:]*uy.vector()[:]))
    
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
    w.vector()[:] = w.vector()[:]/np.max(w.vector()[:])

    return w

def quality_measure(x,y):
    
    ## Compute mesh Skewness
    Q = FunctionSpace(mesh,'DG',0)
    q = Function(Q)
    
    grad_x = project(grad(x),VectorFunctionSpace(mesh_c,'DG',0))
    grad_y = project(grad(y),VectorFunctionSpace(mesh_c,'DG',0))
     
    for c in cells(mesh):       
        
        ## evaluate gradient at the midpoint  
        v1 = grad_x(c.midpoint().x(),c.midpoint().y())
        v2 = grad_y(c.midpoint().x(),c.midpoint().y()) 
       
        Gmatrix = np.array([v1,v2])
        eigval,eigvec = np.linalg.eig(Gmatrix)
        lambda_1, lambda_2 = abs(eigval)
        
        offset = 1e-16
        lambda_1 += offset
        lambda_2 += offset
        
        q.vector()[c.index()] = (lambda_1/lambda_2 + lambda_2/lambda_1)/2.0
    
    return q


def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/ln(sqrt(dof[i+1]/dof[i]))

    return rate

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


monitor_str = 'curvature'
parameters['allow_extrapolation'] = True

tol = 1e-16
dt = 1e-2
tau = 1.0
itmax = 15
beta = 1e-4

## Create L-shaped domain 
rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2


Ns = 2**np.arange(5,10)
Energy_norm = np.zeros(Ns.shape[0])
dof = np.zeros(Ns.shape[0])    

for i,N in enumerate(Ns):
    
    print('iteration n°: ' + str(i))
    mesh = mshr.generate_mesh(geometry, N)   # physical mesh 
 
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    CG3 = FunctionSpace(mesh, "CG", 3)
   
    u_expr = Expression_uexact(degree=5)    
    gradu_expr = Expression_grad_uexact(degree=5)  

    u_exact = interpolate(u_expr,CG3)
    
    ## Solve Poisson Equation
    u = solve_poisson(u_expr)
    
    mesh_c = mshr.generate_mesh(geometry, N) # computational mesh

    X = FunctionSpace(mesh_c,'CG',1) 
    CG1_c = FunctionSpace(mesh_c,'CG',1) 
    DG0_c = FunctionSpace(mesh_c,'DG',0) 
    
    ## monitor function 
    w_old = monitor()
    
    ## Smooth monitor function 
    #w = smoothing(w_old,beta)
    
    ## Solve MMPDE based on Winslow's equation
    
    x_trial = TrialFunction(X)
    y_trial = TrialFunction(X)
    
    x_test = TestFunction(X)
    y_test = TestFunction(X)
    
    x_old = Function(X)
    y_old = Function(X)
    
    x_old.interpolate(Expression('x[0]',degree=1))
    y_old.interpolate(Expression('x[1]',degree=1))

    x_new = Function(X)
    y_new = Function(X)
    
    x0 = Function(X)
    x1 = Function(X)
    
    y0 = Function(X)
    y1 = Function(X)
    
    Ax = assemble(x_trial*x_test*dx(domain = mesh_c))
    Ay = assemble(y_trial*y_test*dx(domain = mesh_c))
    
   
    it = 0
    err = np.zeros(itmax)
    
    bc_x = [DirichletBC(X,0.0,boundary_2),DirichletBC(X,1.0,boundary_4),DirichletBC(X,-1.0,boundary_6)]
    bc_y = [DirichletBC(X,-1.0,boundary_1),DirichletBC(X,0.0,boundary_3),DirichletBC(X,1.0,boundary_5)]
     
  
    
    while (it < itmax):
        
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
      
      solve(a==L,x_new,bc_x)
      
      
      K1 = A*y_trial*y_test*dx
      K2 = w*(-(alpha_1*y_test).dx(0)*y_trial.dx(0) + ((alpha_2*y_test).dx(1)*y_trial.dx(0) + 
                                                       (alpha_2*y_test).dx(0)*y_trial.dx(1)) - 
              (alpha_3*y_test).dx(1)*y_trial.dx(1))*dx
      
      a = K1 - dt*K2
      L = A*y_old*y_test*dx
      
      solve(a==L,y_new,bc_y)
      
      x_old = x_new
      y_old = y_new
      
      # update mesh coordinates 
      mesh.coordinates()[:,0] = x_new.compute_vertex_values()
      mesh.coordinates()[:,1] = y_new.compute_vertex_values()
      
      
      mesh.bounding_box_tree().build(mesh)
      mesh_c.bounding_box_tree().build(mesh_c)
      
      u = solve_poisson(u_expr)
      
      w = monitor()
      w = smoothing(w,beta)
      
      #err[it] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 

      
      it += 1      
        
    dof[i] = CG1.dim()
    Energy_norm[i] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 5})) 
      


# For CG1 convergence rate in H1 semi-norm is expected to be 1 
rate = conv_rate(dof,Energy_norm)
fig, ax = plt.subplots()
ax.plot(dof,Energy_norm,linestyle = '-.',marker = 'o',label = 'rate: %.4g' %rate[-1])
ax.plot(dof_movmesh_aposteriori,H10_movmesh_aposteriori,linestyle = '-.',marker = 'o',label = 'blah')
ax.set_xlabel('dof')
ax.set_ylabel('H10 error')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'best')           
     

np.save('H10_movmesh_' + monitor_str + '.npy',Energy_norm)
np.save('dof_movmesh_' + monitor_str + '.npy',dof)
np.save('rate_movmesh_' + monitor_str + '.npy',rate)
