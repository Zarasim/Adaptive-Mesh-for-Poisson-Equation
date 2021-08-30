#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

@author: simo94

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


def refinement(mesh,beta,type_ref,*args):
    
    dx = Measure('dx',domain = mesh)
    dS = Measure('dS',domain = mesh)

    w = TestFunction(DG0)
    cell_residual = Function(DG0)
    n = FacetNormal(mesh)
    
    indicator_exp = Expression_aposteriori(mesh,beta,degree=0)
    hk = CellDiameter(mesh)
    K = CellVolume(mesh)

    # For f = 0 and p=1 the first term disappear 
    monitor_tensor = avg(w)*(avg(hk**(3-2*indicator_exp))*jump(grad(u),n)**2 +  avg(hk**(1-2*indicator_exp))*(jump(u,n)[0]**2 + jump(u,n)[1]**2))*dS(mesh)
    assemble(monitor_tensor, tensor=cell_residual.vector())

 
    # Compute equidistributing indicator
    es_residual = sum(cell_residual.vector()[:])/(mesh.num_cells())
    
    # Mark cells for refinement
    cell_markers = MeshFunction('bool',mesh,mesh.topology().dim())   
    
    # Maximum strategy 
    if type_ref == 'MS':
        
        ref_ratio = 0.1
        gamma_0 = sorted(cell_residual.vector()[:],reverse = True)[int(mesh.num_cells()*ref_ratio)]
        gamma_0 = MPI.max(mesh.mpi_comm(),gamma_0)

        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[c.index()] > gamma_0
        
    # Equidistribution strategy
    elif  type_ref == 'ES':
        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[c.index()] > es_residual*tol

    # Guaranteed error reduction strategy (Dorfler')
    elif type_ref == 'GERS':
        
        
        theta_star = args[0]
        nu = args[1]
        est_sum2_marked = 0;
        threshold = (1 - theta_star)**2 * sum_residual;
        gamma = 1;
        max_est = np.max(cell_residual.vector()[:])
        
        while (est_sum2_marked < threshold):
            gamma = gamma - nu   
            v = []
            for c in cells(mesh):
                ineq_check = cell_residual.vector()[c.index()] > gamma*max_est
                if ineq_check:
                    cell_markers[c.index()] = cell_residual.vector()[c.index()] > gamma*max_est
                    v.append(c.index())
                    
            est_sum2_marked = sum(cell_residual.vector()[v])
    
    # Refine mesh 
    mesh = refine(mesh,cell_markers)
            
    return mesh,cell_residual


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
        if (r < 1e-5):
            print('tu(%s) = %g' %(x, w(x)))
            w_1d.append(w(x))
            dist.append(r)
            
    w_1d = np.array(w_1d)
    dist = np.array(dist)
    
    w_1d[::-1].sort() 
    dist.sort()
    return w_1d,dist

   
def conv_rate(dof,err):

    'Compute convergence rate '    
    
    l = dof.shape[0]
    rate = np.zeros(l-1)
       
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/(ln(dof[i+1]/dof[i]))

    return rate

output = 1
beta = 0.99


#rectangle1 = mshr.Rectangle(Point(-1.0,0.0),Point(1.0,1.0))
#rectangle2 = mshr.Rectangle(Point(-1.0,0.0),Point(0.0,-1.0))
#geometry = rectangle1 + rectangle2
#mesh = mshr.generate_mesh(geometry, 10) 
#mesh.bounding_box_tree().build(mesh)


mesh = Mesh('ell_mesh.xml')
mesh.rotate(-90)
mesh.coordinates()[:] = mesh.coordinates()[:]/2 - np.array([1.0,0.6923076923076923])

# uniform criss-cross
#mesh =  Mesh('mesh_uniform/mesh_uniform_771.xml.gz')


n_ref = 17
Energy_norm = np.zeros(n_ref)
L2_norm = np.zeros(n_ref)
dof = np.zeros(n_ref)
mu_vec = np.zeros(n_ref)
q_vec = np.zeros(n_ref)

## Pvd file

if output:
    file_w = File('Paraview/h-ref/w_'+ str(beta) +'.pvd')
    file_u = File('Paraview/h-ref/u_'+ str(beta) +'.pvd')
    file_q = File('Paraview/h-ref/q_'+ str(beta) +'.pvd')
    file_mu = File('Paraview/h-ref/mu_'+ str(beta) +'.pvd')

it = 0

while it < n_ref:

  print('iteration n° ',it)
  DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
  V = FunctionSpace(mesh, "DG", 1) # function space for solution u

  
  #if it > 0:
  # print w at first iteration
  if it ==1:
      w_1d,dist = monitor_1d(mesh,w)
 

  u_exp = Expression_u(degree=5)
  gradu_exp = Expression_grad(degree=5)
  f = Constant('0.0')
  u = solve_poisson(u_exp)
  mesh.bounding_box_tree().build(mesh)
  
  mesh,w = refinement(mesh,beta,'MS')
    
  q = mesh_condition(mesh)
  mu = shape_regularity(mesh)
  L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh))) 
  dof[it] = V.dim()



  if output:
      
      w.rename('w','w')
      mu.rename('mu','mu')
      q.rename('q','q')
      u.rename('u','u')
      
      file_w << w,it
      file_mu << mu,it
      file_q << q,it
      file_u << u,it
  
  q_vec[it] = np.max(q.vector()[:])
  mu_vec[it] = np.min(mu.vector()[:]) 
  it += 1      
  

#rate = conv_rate(dof,L2_norm)
#,label = 'rate: %.4g' %np.mean(rate[-5:])
#fig, ax = plt.subplots()
#ax.plot(dof,q_vec,linestyle = '-.',marker = 'o')
#ax.set_xlabel('dof')
#ax.set_ylabel('L2 error')
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.legend(loc = 'best')       

#np.save('Data/h_ref/L2_href_' + str(beta) +'.npy',L2_norm)
#np.save('Data/h_ref/dof_href_' + str(beta) +'.npy',dof)
#np.save('Data/h_ref/rate_href_' + str(beta) +'.npy',rate)
#np.save('Data/h_ref/q_' + str(beta) +'.npy',skewness)
#np.save('Data/h_ref/q.npy',q_vec)
#np.save('Data/h_ref/mu.npy',mu_vec)
np.save('Data/h_ref/monitor.npy',w_1d)
np.save('Data/h_ref/dist.npy',dist)