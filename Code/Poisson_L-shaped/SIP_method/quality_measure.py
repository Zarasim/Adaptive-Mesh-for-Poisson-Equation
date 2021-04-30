#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:26:13 2021

Define all functions to evaluate the quality of a resulting mesh 

@author: simone
"""


from dolfin import *
import numpy as np


def mesh_condition(mesh):
    
    DG0 = FunctionSpace(mesh,"DG",0)
    q = Function(DG0)
    
    D = mesh.topology().dim()
    mesh.init(D-1,D) # Build connectivity between facets and cells
    
    for f in facets(mesh):
        idxs = f.entities(D)
        if len(idxs) == 2:
            cell1 = Cell(mesh,idxs[0])
            cell2 = Cell(mesh,idxs[1])
            
            hk1 = cell1.h()
            hk2 = cell2.h()
            
            val = 2*abs(hk1-hk2)/(hk1+hk2)
            if val > q.vector()[idxs[0]]: 
                q.vector()[idxs[0]] = val
            
            if val > q.vector()[idxs[1]]: 
                q.vector()[idxs[1]] = val
            
    return q


def skewness(mesh_c,mesh,x,y):
    
    # Compute mesh skewness given by 1/2*(sigma1/sigma2 + sigma2/sigma1)
    # sigma1,sigma2 are the eigenvalues of the Jacobian matrix 
    # x,y are the coordinates in the physical mesh as function of computational mesh
    DG0 = FunctionSpace(mesh,"DG",0)
    grad_x = project(grad(x),VectorFunctionSpace(mesh_c,'DG',0))
    grad_y = project(grad(y),VectorFunctionSpace(mesh_c,'DG',0))
    Q = Function(DG0)
    
    for c in cells(mesh):       
        
        v1 = grad_x(c.midpoint().x(),c.midpoint().y())
        v2 = grad_y(c.midpoint().x(),c.midpoint().y())
        
        Gmatrix = np.array([v1,v2])
        eigval,eigvec = np.linalg.eig(Gmatrix)
        lambda_1, lambda_2 = abs(eigval)
        
        offset = 1e-16
        lambda_1 += offset
        lambda_2 += offset
        
        Q.vector()[c.index()] = (lambda_1/lambda_2 + lambda_2/lambda_1)/2.0
        
    return Q



def shape_regularity(mesh):
    
    ## Compute mesh Skewness
     
    DG0 = FunctionSpace(mesh,"DG",0)
    mu = Function(DG0)
    
    for c in cells(mesh):       
        
        pk = c.inradius()
        hk = c.h()
        
        mu.vector()[c.index()] = pk/hk
    
    return mu
