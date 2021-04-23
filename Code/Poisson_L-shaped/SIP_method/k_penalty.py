#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:05:38 2021

@author: simone
"""


def penalty_parameter(mesh):
    
    '''
    The interior penalty parameter k must be chosen such that 
    
    k > 4 max(rho(Sk))
    
    1) For each cell compute barycentric coordinates given by
    
    lambda_j = (1/2|k|)*dot(( x - p_j+1),perp(p_j-1 - p_j+1))
    
    2) Compute grad(lambda_j) = (1/2|k|)*perp(p_j-1 + p_j+1)
    3) Assemble matrix Sk
    4) Compute spectral radius for each cell 
    5) Find maximum spectral radius 
    
    '''
    # loop over each cell of the mesh 
    
    ncells = mesh.num_cells()
    rho = np.zeros(ncells)
    
    for idx_cell in range(ncells):
        
        cell = Cell(mesh,idx_cell) 
        
        # get coordinates of the verices 
        coords = np.array(cell.get_vertex_coordinates()).reshape(3,2)
        
        # get coord in anti-clockwise order 
        coords = AntiClockOrder(coords)
        
        # compute area of cell
        vol = 1/(2*cell.volume())
        
        S_k = np.zeros([3,3])
        
        for i in range(3):
            
            p = coords[i-1,:] - coords[((i+1)%3),:]
            p = np.array([-p[1],p[0]])
            grad_1 = vol*p
        
            for j in range(3):
                
                p = coords[i-1,:] - coords[((i+1)%3),:]
                p = np.array([-p[1],p[0]])
                grad_2 = vol*p
                
                S_k[i,j] = np.dot(grad_1,grad_2)
                
        
        lambdas,v = np.linalg.eig(S_k)
        rho[idx_cell] = max(abs(lambdas))
        
    
    k_penalty = 4*np.max(rho) + 1
        
    return k_penalty