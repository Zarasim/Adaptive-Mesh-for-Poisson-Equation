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
#parameters["form_compiler"]["quadrature_degree"] =  3
set_log_active(False) # handling of log messages, warnings and errors.


# Expression for exact solution 
class Expression_u(UserExpression):
    
    def __init__(self,omega,corners,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.omega = omega
        self.corners = corners
    
    def eval(self, value, xv):
        
        ## find closest corner from x and evaluate expression 
        r = 100
        corner = [0.0,0.0]
        
        for c in self.corners:
            min_d = np.linalg.norm(xv-c)
            if min_d < r:
                r = min_d
                corner = c
        
        idx = self.corners.index(corner)
        x = xv - corner 
    
        if (r > 1.0):
            value[0] == 0.0
            return
        
        theta = math.atan2(abs(x[1]),abs(x[0]))
        
        if idx == 0:
            
            if x[0] < 0 and x[1] > 0:
                theta = pi - theta
                
            elif x[0] <= 0 and x[1] <= 0:
                theta = pi + theta
                
            if (r == 0.0) or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] >= 0):           
                value[0] = 0.0
                return
            
        elif idx == 1:
            
            if x[0] < 0 and x[1] > 0:
                theta = pi/2 - theta
            
            elif x[0] <= 0 and x[1] <= 0:
                theta = pi/2 + theta
                
            elif x[0] > 0 and x[1] < 0:
              theta = 3/2*pi-theta  
            
            if (r == 0.0) or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] >= 0):           
                value[0] = 0.0
                return
                
        elif idx == 2:
            
            if x[0] >= 0 and x[1] <= 0:
                theta = pi - theta
                            
            elif x[0] > 0 and x[1] >= 0:
                theta = pi + theta
                          
            if (r == 0.0) or (x[0] == 0 and x[1] >= 0) or (x[1] == 0 and x[0] <= 0):
                value[0] = 0.0
                return
        
        else:
            
             if x[0] >= 0 and x[1] <= 0:
                 theta = pi/2 - theta
        
             elif x[0] > 0 and x[1] > 0:
                theta = pi/2 + math.atan2(abs(x[1]),abs(x[0]))
                        
             elif x[0] < 0 and x[1] > 0:
                theta = 3*pi/2 - math.atan2(abs(x[1]),abs(x[0]))
                      
             if (r == 0.0) or (x[0] == 0 and x[1] <= 0) or (x[1] == 0 and x[0] <= 0):
                    value[0] = 0.0     
                    return
        
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
    

beta = 0.8
def monitor(mesh,u,type_norm,*args):
    
    w = TestFunction(DG0)
    cell_residual = Function(DG0)
    n = FacetNormal(mesh)
    area_cell = Expression_cell(mesh,degree=0)        
    area_cell_func = interpolate(area_cell,DG0)
    hk = CellDiameter(mesh)
    p = args[0]
    
    
    if type_norm == 'Linfty':
       
        # find the minimum cell diameter over all hk
        
        mincell = MinCellEdgeLength(mesh)
        l_hd = ln(1/mincell)**2
        
        # Iterate thorugh every cell and evaluate the maximum looking at the adjacent ones for jump terms 
        monitor_tensor = avg(l_hd)*avg(w)*pow(avg(hk)*jump(grad(u),n),p)/avg(hk)*dS(mesh) \
        + avg(l_hd)*avg(w)*pow(jump(u,n)[0] + jump(u,n)[1],p)/avg(hk)*dS(mesh) \
        + l_hd*w*pow(u_exp - u,p)/hk*ds(mesh) 
    
        assemble(monitor_tensor, tensor=cell_residual.vector())

        
    else:
        
        indicator_exp = Expression_aposteriori(mesh,beta,degree=0)      
        monitor_tensor = (avg(w)*(avg(hk**(3-2*indicator_exp))*jump(grad(u),n)**2 \
                         + avg(hk**(1-2*indicator_exp))*(jump(u,n)[0]**2 + jump(u,n)[1]**2)))*dS(mesh)    
        assemble(monitor_tensor, tensor=cell_residual.vector())

    
    cell_residual.vector()[:] = np.power(cell_residual.vector()[:],1.0/p)
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


#num=30
#gamma_vec = np.linspace(0.0,0.9,num)[10:]
gamma_vec = np.array([0.5])
n_ref = 3

# Solve Poisson Equation
L2_norm = np.zeros(n_ref)
Linfty_norm = np.zeros(n_ref)


output = 1
p = 5
corners = [[3.0,1.0],[2.0,4.0],[-6.0,5.0],[-5.0,3.0]]


for it in range(n_ref):
    
   # compute error for the OT mesh with fixed dof 
   print('Iteration for ref° : ', it)
   
   if output:
       file_u = File('Paraview/OT_priori/u.pvd')

   string_mesh = 'Mesh/0.5/mesh_T_' + str(it) + '.xml.gz'
   mesh = Mesh(string_mesh)    
   coords = mesh.coordinates()[:]
   
   DG0 = FunctionSpace(mesh, "DG", 0) # define a-posteriori monitor function 
   DG1 = FunctionSpace(mesh, "DG", 1) 
   CG1 = FunctionSpace(mesh,"CG",5)
   V = FunctionSpace(mesh, "DG", 1) # function space for solution u
   
   
   omega = 3.0/2.0*pi
   u_exp = Expression_u(omega,corners,degree=5)
   f = Constant('0.0')
      
   u = solve_poisson(u_exp)
   plt.figure()
   plot(u)
   
   mesh.bounding_box_tree().build(mesh)   
   
   #D = mesh.topology().dim()
   #mesh.init(D-1,D) # Build connectivity between facets and cells      

   if output:
      u.rename('u','u')    
      file_u << u,it
          
   L2_norm[it] = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh)))
   
#   maxErr = 0
#   for i,x in enumerate(coords):
#       err = abs(u_exp.eval_at_point(x) - u(x))
#       if err > maxErr:
#           maxErr = err
#       
#   Linfty_norm[it] = maxErr
        
   
   
np.save('Data/L2_'  + str(np.round(gamma, 2)) + '.npy',L2_norm)
#np.save('Data/Linfty_'  + str(np.round(gamma, 2)) + '.npy',Linfty_norm)
#
#
#dict = {'gamma': gamma_vec, 'error_L2': L2_norm, 'error_Linfty': Linfty_norm }   
#df = pd.DataFrame(dict) 
#df.to_csv('Data/error_gamma.csv',index=False) 
