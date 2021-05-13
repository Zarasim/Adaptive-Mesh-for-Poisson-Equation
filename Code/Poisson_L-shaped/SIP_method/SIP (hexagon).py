#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

using a discontinuous Galerkin formulation (symmetric interior penalty method).

"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import mshr 
import math
from quality_measure import *

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.

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
        
class Expression_grad(UserExpression):
    
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
    
    
class Expression_aposteriori(UserExpression):
    
    def __init__(self, mesh,beta, **kwargs):
        self.mesh = mesh
        self.beta = beta
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        #n = cell.normal(ufc_cell.local_facet)
        
        if cell.contains(Point(0.0,0.0)):
           
            values[0] = np.float(self.beta)
       
        else:
            
           values[0] = np.float(0.0)
        
    def value_shape(self):
        return ()
    
    
def solve_poisson(u_exp):
    
    n = FacetNormal(mesh)
    h =  CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2
     
    # Define parameters
    #C_sigma = 20
    k_penalty = 20
    
    v = TestFunction(V)
    u = TrialFunction(V)
    
    # Define variational problem
    a = dot(grad(v), grad(u))*dx(mesh) \
       - dot(avg(grad(v)), jump(u, n))*dS(mesh) \
       - dot(jump(v, n), avg(grad(u)))*dS(mesh) \
       + (k_penalty/h_avg)*dot(jump(v, n), jump(u, n))*dS(mesh) \
       - dot(grad(v), u*n)*ds(mesh) \
       - dot(v*n, grad(u))*ds(mesh) \
       + (k_penalty/h)*v*u*ds(mesh)
    
    L = v*f*dx(mesh) - u_exp*dot(grad(v), n)*ds(mesh) + (k_penalty/h)*u_exp*v*ds(mesh)
 
    u = Function(V)
    
    solve(a==L,u)
    
    return u
    
    
def monitor(mesh,u,beta,type_monitor):
    
    '''
    monitor function for a-posteriori estimate of SIPDG method in L2 norm 
    
    '''
    
    dx = Measure('dx',domain = mesh)
    dS = Measure('dS',domain = mesh)  
    area = assemble(Constant(1.0)*dx(mesh))
    
    if type_monitor == 'gradient':
        
        grad_u = project(grad(u), VectorFunctionSpace(mesh, 'CG', 1))
        
        ux = project(grad(u)[0],CG1)
        uy = project(grad(u)[1],CG1)
        
        #alpha = np.power((1/area)*assemble(pow(dot(grad_u,grad_u),0.5)*dx(mesh)),2) 
        
        w = Function(X)
        w.vector()[:] = np.sqrt(1.0 + 10000*(ux.vector()[dof_P_to_C]**2 + uy.vector()[dof_P_to_C]**2))
    
        return w
            
    elif type_monitor == 'curvature':
        
        grad_u = project(grad(u),FunctionSpace(mesh,'RT',1)) 
        laplacian = project(div(grad_u),CG1) 
       
        #alpha = pow((1/area)*assemble(pow(abs(laplacian),0.5)*dx(mesh)),2)
        
        w = Function(X)
        w.vector()[:] = np.power(1.0 + 10*abs(laplacian.vector()[dof_P_to_C]),1.0/2.0)   
    
        return w
    
    elif type_monitor == 'a-posteriori':

        w = TestFunction(DG0)
        cell_residual = Function(DG0)
        n = FacetNormal(mesh)
    
        indicator_exp = Expression_aposteriori(mesh,beta,degree=0)
        hk = CellDiameter(mesh)
    
        # For f = 0 and p=1 the first term disappear 
        monitor_tensor = avg(w)*(avg(hk**(3-2*indicator_exp))*jump(grad(u),n)**2 +  avg(hk**(1-2*indicator_exp))*(jump(u,n)[0]**2 + jump(u,n)[1]**2))*dS(mesh)
        assemble(monitor_tensor, tensor=cell_residual.vector())

        #area = assemble(Constant(1.0)*dx(mesh))        
        monitor = interpolate(cell_residual,CG1)
        monitor_func = monitor.copy(deepcopy = True)
        # rescale the monitor function  
        monitor.vector()[:] = monitor.vector()[:]/np.max(monitor.vector()[:])
    
        w = Function(X)    
        
        #alpha = pow((1/area)*assemble(np.power(monitor,0.5)*dx(mesh)),2)
        #w.vector()[:] = np.sqrt(1 + 10000*monitor.vector()[dof_P_to_C])   
        w.vector()[:] = np.sqrt(1 + 100000*monitor.vector()[dof_P_to_C])
        
        return w,monitor_func


def smoothing(w,diff):
    
    w_test = TestFunction(X)
    w_trial = TrialFunction(X) 
    
    dx = Measure('dx',domain = mesh_c)
    a = w_test*w_trial*dx + diff*inner(grad(w_test),grad(w_trial))*dx 
    L = w_test*w*dx

    w = Function(X)
    solve(a==L,w)
    
    return w


def equidistribute(x,rho):
    
    '''
        input parameters:
            x: coordinates in physical domain 
            w: monitor function
            
        output parameters:
            y: equidistributed coordinates 
            
        The algorithm works only if the y coords are defined in increasing order
        
    '''

    # number of mesh points counted from 0 ton nx-1
    nx = x.shape[0]
    
    y = x.copy()
    y = np.array(sorted(y))
    
    dof_to_sort = [np.where(x == y[i])[0].tolist() for i in range(nx)]
    dof_to_sort = np.hstack(dof_to_sort)
    
    sort_to_dof = [np.where(y == x[i])[0].tolist() for i in range(nx)]
    sort_to_dof = np.hstack(sort_to_dof)
    
    y_new = y.copy()
    
    
    rho[:] = rho[:]/np.max(rho[:]) # scale
    rho = rho[dof_to_sort]
    
    II = nx - 1 
    JJ = nx - 1
    
    # Create vector of integrals with nx entries 
    intMi = np.zeros(nx)
    
    # compute each integral using trapezoidal rule
    intMi[1:] = 0.5*(rho[1:] + rho[:-1])*np.diff(y)
    
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
        
        Xl = y[jj]
        Xr = y[jj+1]
        Ml = rho[jj]
        Mr = rho[jj+1]
        
        Target_loc = Target - intM[jj]
        
        mx = (Mr - Ml)/(Xr - Xl)
        
        y_new[ii] = Xl + 2*Target_loc/(Ml + np.sqrt(Ml**2 + 2*mx*Target_loc))
       
        
    return y_new[sort_to_dof]
    


def boundary_mesh(x_old,y_old,w,n_iter = 20):
   
    yb = y_old.vector()[dofs_1]
    wb = w.vector()[dofs_1]
    
    xt = x_old.vector()[dofs_2]
    wt = w.vector()[dofs_2]
    
    for it in range(n_iter):
        yb = equidistribute(yb,wb)
        xt = equidistribute(xt,wt)

    X_boundary.vector()[dofs_2] = xt  
    Y_boundary.vector()[dofs_1] = yb   
    
    return X_boundary,Y_boundary



def monitor_1d(mesh,w):
    
    ## return array of r,values points to fit successively
    w_1d = []
    dist = []
    
    tol = 1e-12
    # find dof associated with the points closest to the corner 
    vertex_values = u.compute_vertex_values()
    coordinates = mesh.coordinates()
    
    for i, x in enumerate(coordinates):
        r = np.linalg.norm(x)
        if (r < 0.1):
            print('tu(%s) = %g' %(x, w(x)))
            w_1d.append(w(x))
            dist.append(r)
            
    w_1d = np.array(w_1d)
    dist = np.array(dist)
    
    w_1d[::-1].sort() 
    dist.sort()
    return w_1d,dist


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



tol = 1e-16
tau = 1.0
itmax = 500
beta = 0.99
output_file = 0
dt = 1e-2
delta = 1e-3
#delta = 1e-3  except beta=0.99

type_monitor = 'a-posteriori'

rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
geometry = rectangle1 + rectangle2
 
N = 2**np.array([5])
#N = 10
#N = 120

f = Constant(0.0)
    
# Create mesh and define function space
mesh = mshr.generate_mesh(geometry,N)  # physical mesh 

DG0 = FunctionSpace(mesh, 'DG', 0)
CG1 = FunctionSpace(mesh, 'CG', 1)
V = FunctionSpace(mesh, 'DG', 1)


u_exp = Expression_u(degree=5)    
grad_exp = Expression_grad(degree=5)


u = solve_poisson(u_exp)  



if output_file:    
    
    if type_monitor == 'a-posteriori':
        file_w = File('Paraview/L-shaped/r-adaptive/monitor.pvd')
        #file_mu = File('Paraview/r-adaptive/'  + str(type_monitor) +  '_'  + str(beta) + '_dof_' + str(V.dim()) + '/mu.pvd')
        #file_q = File('Paraview/r-adaptive/'  + str(type_monitor) +  '_'  + str(beta) + '_dof_' + str(V.dim()) + '/q.pvd')
        #file_xi = File('Paraview/L-shaped/r-adaptive/xi.pvd')
        #file_nu = File('Paraview/L-shaped/r-adaptive/nu.pvd')
#    else:
#        file_u = File('Paraview/L-shaped/r-adaptive/' + str(type_monitor) + '_dof_' + str(V.dim()) + '/poisson_SIP.pvd')
#        file_mu = File('Paraview/L-shaped/r-adaptive/'  + str(type_monitor) + '_dof_' + str(V.dim()) + '/mu.pvd')
#        #file_w = File('Paraview/L-shaped/r-adaptive/'  + str(type_monitor) + '_dof_' + str(V.dim()) + '/w.pvd')
#        #file_xi = File('Paraview/L-shaped/r-adaptive/xi.pvd')
#        #file_nu = File('Paraview/L-shaped/r-adaptive/nu.pvd')
#        
#    file_xi << x_old
#    file_nu << y_old
#    
#    u.rename('u','u')
#   # w.rename('w','w')
#    
#    file_u << u,it
#    file_w << w,it


mu_vec = []
q_vec = []


##########  Solve Laplace  ###########

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
   
editor.close()

X = FunctionSpace(mesh_c,'CG',1) 
DG0_c = FunctionSpace(mesh_c,'DG',0) 

coords_C = X.tabulate_dof_coordinates()
dofmap_C = X.dofmap()

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
w,monitor_func = monitor(mesh,u,beta,type_monitor)
w = smoothing(w,delta)
## Solve MMPDE based on Winslow's equation

x_trial = TrialFunction(X)
y_trial = TrialFunction(X)

x_test = TestFunction(X)
y_test = TestFunction(X)

x_old = Function(X)
y_old = Function(X)

xp = interpolate(Expression('x[0]',degree=1),CG1)
yp = interpolate(Expression('x[1]',degree=1),CG1)

x_old.vector()[:] = xp.vector()[dof_P_to_C]
y_old.vector()[:] = yp.vector()[dof_P_to_C]

dofs_1 = np.intersect1d(np.where(x_old.vector() == 0.0)[0],np.where(y_old.vector() <= 0.0)[0])    
dofs_2 = np.intersect1d(np.where(y_old.vector() == 0.0)[0],np.where(x_old.vector() >= 0.0)[0])

x_new = Function(X)
y_new = Function(X)

x0 = Function(X)
x1 = Function(X)

y0 = Function(X)
y1 = Function(X)

Ax = assemble(x_trial*x_test*dx(domain = mesh_c))
Ay = assemble(y_trial*y_test*dx(domain = mesh_c))

x_b = x_old.copy(deepcopy = True)
y_b = y_old.copy(deepcopy = True)

X_boundary = x_old.copy()
Y_boundary = y_old.copy()
   
it = 0
err = np.zeros(itmax)
skewness = np.zeros(itmax)
diff = np.zeros(itmax)


#if output_file:    
#    
#    if type_monitor == 'a-posteriori':
        #file_u = File('Paraview/r-adaptive/' + str(type_monitor) +  '_' + str(beta) + '_dof_' + str(V.dim()) + '/u.pvd')
        #file_q = File('Paraview/r-adaptive/'  + str(type_monitor) +  '_'  + str(beta) + '_dof_' + str(V.dim()) + '/q.pvd')
        #file_mu = File('Paraview/r-adaptive/'  + str(type_monitor) +  '_'  + str(beta) + '_dof_' + str(V.dim()) + '/mu.pvd')
        #file_w = File('Paraview/r-adaptive/'  + str(type_monitor) +  '_'  + str(beta) + '_dof_' + str(V.dim()) + '/w.pvd')
       # file_xi = File('Paraview/L-shaped/r-adaptive/xi.pvd')
       # file_nu = File('Paraview/L-shaped/r-adaptive/nu.pvd')
    #else:
        #file_u = File('Paraview/L-shaped/r-adaptive/' + str(type_monitor) + '_dof_' + str(V.dim()) + '/poisson_SIP.pvd')
     #   file_q = File('Paraview/L-shaped/r-adaptive/'  + str(type_monitor) + '_dof_' + str(V.dim()) + '/q.pvd')
        #file_w = File('Paraview/L-shaped/r-adaptive/'  + str(type_monitor) + '_dof_' + str(V.dim()) + '/w.pvd')
        #file_xi = File('Paraview/L-shaped/r-adaptive/xi.pvd')
        #file_nu = File('Paraview/L-shaped/r-adaptive/nu.pvd')
#        nu.rename('nu','nu')
#        file_xi << x_old
#        file_nu << y_old
    
   # u.rename('u','u')
   # w.rename('w','w')
   # file_w << w,it
   # file_u << u,it
#
#if output_file:    
#    
#    if type_monitor == 'a-posteriori':
        
X_boundary,Y_boundary = boundary_mesh(x_old,y_old,w)
bc_x = [DirichletBC(X,X_boundary,'on_boundary'),DirichletBC(X,0.0,boundary_2c),DirichletBC(X,1.0,boundary_4c),DirichletBC(X,-1.0,boundary_6c)]
bc_y = [DirichletBC(X,Y_boundary,'on_boundary'),DirichletBC(X,-1.0,boundary_1c),DirichletBC(X,0.0,boundary_3c),DirichletBC(X,1.0,boundary_5c)] 

while (it < itmax):
    
   
  print('   iteration MMPDE: ' + str(it))
  
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
  
  solve(a==L,x_new,bc_x)#solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})
  
  
  K1 = A*y_trial*y_test*dx
  K2 = w*(-(alpha_1*y_test).dx(0)*y_trial.dx(0) + ((alpha_2*y_test).dx(1)*y_trial.dx(0) + 
                                                   (alpha_2*y_test).dx(0)*y_trial.dx(1)) - 
          (alpha_3*y_test).dx(1)*y_trial.dx(1))*dx
  
  a = K1 - dt*K2
  L = A*y_old*y_test*dx
  
  solve(a==L,y_new,bc_y)#,solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})
  

  diff[it] = (np.linalg.norm(x_old.vector()[:] - x_new.vector()[:]) + np.linalg.norm(y_old.vector()[:] - y_new.vector()[:]))\
   /(np.linalg.norm(x_old.vector()[:]) + np.linalg.norm(y_old.vector()[:]))      
  

  print('diff:', diff[it])
  
  
  if (diff[it] < 1e-5): #or (shape_regularity[it] < 1e-4):
      break
  
  
  x_old = x_new.copy(deepcopy = True)
  y_old = y_new.copy(deepcopy = True)
  
  xp.vector()[:] = x_new.vector()[dof_C_to_P]
  yp.vector()[:] = y_new.vector()[dof_C_to_P]
  
  # update mesh coordinates 
  mesh.coordinates()[:,0] = xp.compute_vertex_values()
  mesh.coordinates()[:,1] = yp.compute_vertex_values()
  
  mesh.bounding_box_tree().build(mesh)
  mesh_c.bounding_box_tree().build(mesh_c)
  
  u = solve_poisson(u_exp)
  
  w,monitor_func = monitor(mesh,u,beta,type_monitor)
  w = smoothing(w,delta)
  
  err[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh)))
   
      
  q = mesh_condition(mesh)
  mu = shape_regularity(mesh)
  
  if output_file:   
   
#      u.rename('u','u')
#      file_u << u,it
      
      monitor_func.rename('w','w')
      file_w << monitor_func,it
      
#      q.rename('q','q')
#      file_q << q,it
      
  it += 1      


  mu_vec.append(np.min(mu.vector()[:]))
  q_vec.append(np.max(q.vector()[:]))


w_1d,dist = monitor_1d(mesh,monitor_func)

if type_monitor == 'a-posteriori':
#    np.save('Data/r-adaptive/' + str(type_monitor) + '/' +  'L2_r-adaptive_' + str(beta) + '_dof_' + str(V.dim()) + '.npy',err)    
#    np.save('Data/r-adaptive/' + str(type_monitor) + '/' +  'mu_r-adaptive_' + str(beta) + '_dof_'  + str(V.dim()) + '.npy',mu_vec)    
#    np.save('Data/r-adaptive/' + str(type_monitor) + '/' +  'q_r-adaptive_' + str(beta) + '_dof_'  + str(V.dim()) + '.npy',q_vec)    
#    string_mesh = 'mesh_r-adaptive/' +'mesh_' + str(beta) + '_dof_' + str(V.dim()) + '.xml.gz'
#    File(string_mesh) << mesh
    

#fig, ax = plt.subplots()
#ax.plot(err,linestyle = '-',marker = 'o')
#ax.set_xlabel('iteration')
#ax.set_ylabel('L2 error')
#ax.legend(loc = 'best')
#ax.set_yscale('log')
#ax.set_xscale('log')    
    np.save('Data/r-adaptive/monitor.npy',w_1d)
    np.save('Data/r-adaptive/dist.npy',dist)

