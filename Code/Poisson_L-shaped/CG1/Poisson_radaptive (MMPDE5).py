#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

Solve Poisson equation in region with corner singularity 

Model problem discussed in
Kim, Seokchan & Lee, Hyung-Chun. (2016). 
A finite element method for computing accurate solutions for Poisson equations with corner singularities using 
the stress intensity factor. Computers & Mathematics with Applications. 71. 10.1016/j.camwa.2015.12.023. 


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
        
        cell_residual = Function(DG0)
        
        # assume avg diameter close to edge length
        residual = h**2*w*(div(grad(u)))**2*dx + avg(w)*avg(h)*jump(grad(u),n)**2*dS
        assemble(residual, tensor=cell_residual.vector())
        
        cell_residual = interpolate(cell_residual,CG1)
    
        alpha = (1/area)*assemble(cell_residual*dx(mesh))
        
        # w must be interpolated in the computational mesh for integration
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + (1/alpha)*cell_residual.vector()[:])
         
    elif monitor_str == 'curvature':
        
        grad_u = project(grad(u),FunctionSpace(mesh,'RT',1)) 
        laplacian = project(div(grad_u),CG1) 
        laplacian.vector()[:] = laplacian.vector()[:] 
        
        w.vector()[:] = np.sqrt(1.0 + alpha*np.abs(laplacian.vector()[:]))
    
    
    elif monitor_str == 'gradient':
        
        # Project solution gradient in CG1 space 
        grad_u = project(grad(u), VectorFunctionSpace(mesh, 'CG', 1))
        #grad_u.vector()[:] = grad_u.vector()[:]/np.max(grad_u.vector()[:])    
                
        ux = project(grad(u)[0],CG1)
        uy = project(grad(u)[1],CG1)
        
        w = Function(CG1_c)
        w.vector()[:] = np.sqrt(1.0 + alpha*(ux.vector()[:]*ux.vector()[:] + uy.vector()[:]*uy.vector()[:]))
         

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



output = 1
monitor_str = 'a-posteriori'
parameters['allow_extrapolation'] = True

tol = 1e-16
N = 2**4
dt = 5e-3
tau = 1.0
itmax = 10
beta = 1e-6
dxi = 1/N


## Create L-shaped domain 
rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2

    
mesh_c = mshr.generate_mesh(geometry, N) # computational mesh
mesh = mshr.generate_mesh(geometry, N)   # physical mesh 
    
    
DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)
    
    
X = FunctionSpace(mesh_c,'CG',1) 
DG0_c = FunctionSpace(mesh_c,'DG',0) 
    
 
u_expr = Expression_uexact(degree=5)
gradu_expr = Expression_grad_uexact(degree=5)   
    
## Solve Poisson Equation
u = solve_poisson(u_expr)
        
## monitor function 
w_old = monitor()
    
## Smooth monitor function 
w = smoothing(w_old,beta)

## Solve MMPDE based on Winslow's equation
x_trial = TrialFunction(X)
y_trial = TrialFunction(X)

x_test = TestFunction(X)
y_test = TestFunction(X)

x_old = Function(X)
y_old = Function(X)

x_old.interpolate(Expression('x[0]',degree=1))
y_old.interpolate(Expression('x[1]',degree=1))

X_boundary = Function(X)
Y_boundary = Function(X)
    

X_boundary.interpolate(Expression('x[0]',degree=1))
Y_boundary.interpolate(Expression('x[1]',degree=1))


dofs_1  = np.intersect1d(np.where(x_old.vector() == 0.0)[0],np.where(y_old.vector() <= 0.0)[0])    
dofs_2 = np.intersect1d(np.where(y_old.vector() == 0.0)[0],np.where(x_old.vector() >= 0.0)[0])

x_new = Function(X)
y_new = Function(X)

x0 = Function(X)
x1 = Function(X)

y0 = Function(X)
y1 = Function(X)

Ax = assemble(x_trial*x_test*dx(domain = mesh_c))
Ay = assemble(y_trial*y_test*dx(domain = mesh_c))

Energy_norm = np.zeros(itmax+1)
#uex_norm = assemble(inner(grad(u_exact),grad(u_exact))*dx(domain = mesh))
Energy_norm[0] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 2})) 

Q = np.zeros(itmax+1)
q = quality_measure(x_old,y_old)
Q[0] = max(q.vector()[:])

## Pvd file

if output:
    
    file_u = File('Paraview_poisson_radaptive_MMPDE5/u_' + monitor_str + '.pvd')
    file_q = File('Paraview_poisson_radaptive_MMPDE5/q_'+ monitor_str + '.pvd')
    file_w = File('Paraview_poisson_radaptive_MMPDE5/w_'+ monitor_str + '.pvd')
    
    q.rename('q','q')
    u.rename('u','u')
    w.rename('w','w')
    
    file_u << u 
    file_w << w
    file_q << q
    

it = 0
    
while it < itmax:  

    
  # Solve first equation at the boundary
  X_boundary,Y_boundary = boundary_mesh(x_old,y_old,w,tau,dt,dxi)
    
  bc_x = [DirichletBC(X,X_boundary,'on_boundary'),DirichletBC(X,0.0,boundary_2),DirichletBC(X,1.0,boundary_4),DirichletBC(X,-1.0,boundary_6)]
  bc_y = [DirichletBC(X,Y_boundary,'on_boundary'),DirichletBC(X,-1.0,boundary_1),DirichletBC(X,0.0,boundary_3),DirichletBC(X,1.0,boundary_5)]
        
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
  
  # Solve Poisson equation in new mesh 
  u = solve_poisson(u_expr)
  
  w = monitor()
  w = smoothing(w,beta)
  
  it += 1      
  
  q = quality_measure(x_new,y_new)
  Q[it] = max(q.vector()[:])
 
  Energy_norm[it] = np.sqrt(assemble(dot(gradu_expr - grad(u),gradu_expr - grad(u))*dx(mesh),form_compiler_parameters = {"quadrature_degree": 2})) 
  
  if output:
      u.rename('u','u')
      q.rename('q','q')
      w.rename('w','w')
      
      file_u << u 
      file_q << q
      file_w << w



     
plt.figure()
plt.plot(Energy_norm,linestyle = '-.', marker = 'o')
plt.xlabel('iteration')
plt.ylabel('H10 error ' + monitor_str)


plt.figure()
plt.plot(Q,linestyle = '-.', marker = 'o')
plt.xlabel('iteration')
plt.ylabel('local skewness Q ' + monitor_str)



#np.save('H10_'+ str(monitor_str) + '_alpha_' + str(alpha) + '.npy',Energy_norm)
#np.save('Q_'+ str(monitor_str) + '_alpha_' + str(alpha) + '.npy',Q)
