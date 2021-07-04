#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:22:01 2020

First create mest_OT for different gammas and fixed dofs

For a-priori OT mesh plot L2 and L-oo norm as a function of gamma for fixed 
dimension 

@author: simo94
"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt 
import math
from quality_measure import *
import pandas as pd


parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"]     = True  # optimize compiler options 
parameters["form_compiler"]["cpp_optimize"] = True  # optimize code when compiled in c++
set_log_active(False) # handling of log messages, warnings and errors.


# Expression for exact solution 
class Expression_u(UserExpression):
    
    def __init__(self,omega,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.omega = omega
    
    def eval(self, value, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        elif x[0] > 0 and x[1] < 0:
            theta = 2*pi-theta
            
        if r == 0.0:
            value[0] = 0.0
        else:
            value[0] = pow(r,pi/self.omega)*sin(theta*pi/self.omega)
            
    def eval_at_point(self, x):
        
        r =  sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if x[0] < 0 and x[1] > 0:
            theta = pi - theta
            
        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta
            
        elif x[0] > 0 and x[1] < 0:
            theta = 2*pi-theta
            
        if r == 0.0:
            value = 0.0
        else:
            value = pow(r,pi/self.omega)*sin(theta*pi/self.omega)
            
        return value
        
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



class Expression_cell(UserExpression):
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        values[0] = cell.volume()                
        
    def value_shape(self):
        return ()



class Expression_soldiff(UserExpression):
    
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    
    def eval(self, values, x):
               
        tol = 1e-12
        if (near(x[0],0.0,tol) and x[1] <= 0) or (near(x[1],0.0,tol) and x[0] >= 0) or (near(x[0],1.0,tol)) or (near(x[1],1.0,tol)) or (near(x[0],-1.0,tol)) or (near(x[1],-1.0,tol)):
            values[0] = g(x) - u(x)
        else:
            values[0] = 0.0
        
    def value_shape(self):
        return ()


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
    

beta = 0.7
def monitor(mesh,u,type_norm):
    
    w = TestFunction(DG0)
 
    cell_residual = Function(DG0)
    n = FacetNormal(mesh)
    area_cell = Expression_cell(mesh,degree=0)        
    hk = CellDiameter(mesh)

    if type_norm == 'Linfty':
       
        #u_plus = u('+')
        #u_min = u('-')
        #ujump0 = Expression_ujump(mesh,0,u_plus,u_min)
        #ujump1 = Expression_ujump(mesh,1,u_plus,u_min)
        #ujump2 = Expression_ujump(mesh,2,u_plus,u_min)
        
        soldiff = Expression_soldiff(mesh)
        soldiff = interpolate(soldiff,CG3)
        
        # find the minimum cell diameter over all hk
        mincell = MinCellEdgeLength(mesh)
        # The expression can be evaluated separately 
        sold_diff_norm = (ln(1/mincell)**2)*max(abs(soldiff.vector()[:])) 
        
        # Iterate thorugh every cell and evaluate the maximum looking at the adjacent ones for jump terms 
        monitor_tensor = avg((ln(1/mincell)**2))*avg(w)*(abs(avg(hk)*jump(grad(u),n)))/avg(hk)*dS(mesh) \
        + avg((ln(1/mincell)**2))*avg(w)*abs(jump(u,n)[0] + jump(u,n)[1])/avg(hk)*dS(mesh)
        
        #+ avg((ln(1/mincell)**2))*avg(w)*ujump0/avg(hk)*dS(mesh) \
        #+ avg((ln(1/mincell)**2))*avg(w)*ujump1/avg(hk)*dS(mesh) \
        #+ avg((ln(1/mincell)**2))*avg(w)*ujump2/avg(hk)*dS(mesh)
        
        assemble(monitor_tensor, tensor=cell_residual.vector())
        cell_residual.vector()[:] + sold_diff_norm
        #monitor_func = interpolate(cell_residual,'DG0')
        
    else:
        
        indicator_exp = Expression_aposteriori(mesh,beta,degree=0)      
        monitor_tensor = (avg(w)*(avg(hk**(3-2*indicator_exp))*jump(grad(u),n)**2 + \
                                            avg(hk**(1-2*indicator_exp))*(jump(u,n)[0]**2 + jump(u,n)[1]**2)))/avg(hk)*dS(mesh)    
        assemble(monitor_tensor, tensor=cell_residual.vector())
        #monitor_func = interpolate(cell_residual,'DG0')

    return cell_residual 

def monitor_1d(mesh,w):
    
    ## return array of r,values points to fit successively
    w_1d = []
    dist = []
    
    for cell in cells(mesh):
        
        x = np.array([cell.midpoint().x(),cell.midpoint().y()])
        r = np.linalg.norm(x)
        
        if (r < 0.1):
            #print('tu(%s) = %g' %(x, w(x)))
            w_1d.append(w(x))
            dist.append(r)
            
    w_1d = np.array(w_1d)
    dist = np.array(dist)
    
    w_1d[::-1].sort() 
    dist.sort()
    return w_1d,dist



eps = 0.001
omega = 2*pi - eps


num=30
# endpoint is not excluded
gamma_vec = np.linspace(0.0,0.9,num)[15:]

## Solve Poisson Equation
L2_norm = np.zeros(num)
Linfty_norm = np.zeros(num)


# dof = 74880
output = 0

if output:
    file_u = File('Paraview/OT_priori/u.pvd')

for it,gamma in enumerate(gamma_vec):
    
   # compute error for the OT mesh with fixed dof 
   print('Iteration for gamma: ', gamma)
   
   string_mesh = 'Mesh/mesh_OT_' + str(np.round(gamma,2)) + '.xml.gz'
   mesh = Mesh(string_mesh)    
   coords = mesh.coordinates()[:]
   
   DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
   DG1 = FunctionSpace(mesh, "DG", 1) 
   CG1 = FunctionSpace(mesh,"CG",1)
   CG3 = FunctionSpace(mesh, 'CG', 3)
   V = FunctionSpace(mesh, "DG", 1) # function space for solution u

   u_exp = Expression_u(omega,degree=5)
   g = interpolate(u_exp,CG3) 
   f = Constant('0.0')
   
   u = solve_poisson(u_exp)
   mesh.bounding_box_tree().build(mesh)   
   
   if output:
      u.rename('u','u')    
      file_u << u,it
   
   monitor_func_L2 = monitor(mesh,u,'L2')
   monitor_func_Linfty = monitor(mesh,u,'Linfty')

   w_L2,dist = monitor_1d(mesh,monitor_func_L2)
   w_Linfty,dist = monitor_1d(mesh,monitor_func_Linfty)
   

   dict = {'dist': dist, 'measure': w_L2}   
   df = pd.DataFrame(dict) 
   df.to_csv('Data/measure_L2_' + str(round(gamma,2)) + '.csv',index=False) 
    
    
   dict = {'dist': dist, 'measure': w_Linfty}   
   df = pd.DataFrame(dict) 
   df.to_csv('Data/measure_Linfty_' + str(round(gamma,2)) + '.csv',index=False) 
     
          
#   L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh)))
#   
#   maxErr = 0
#   for i,x in enumerate(coords):
#       err = abs(u_exp.eval_at_point(x) - u(x))
#       if err > maxErr:
#           maxErr = err
#       
#   Linfty_norm[it] = maxErr
#        
   
   
#np.save('Data/L2_'  + str(np.round(gamma, 2)) + '.npy',L2_norm)
#np.save('Data/Linfty_'  + str(np.round(gamma, 2)) + '.npy',Linfty_norm)


#dict = {'gamma': gamma_vec, 'error_L2': L2_norm, 'error_Linfty': Linfty_norm }   
#df = pd.DataFrame(dict) 
#df.to_csv('Data/error_gamma.csv',index=False) 