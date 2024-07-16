# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:52:53 2024

@author: hibad
"""
import numpy as np
import cvxpy as cp
import cv2


def outer(a, b):
    a = cp.Expression.cast_to_const(a)  # if a is an Expression, return it unchanged.
    assert a.ndim == 1
    b = cp.Expression.cast_to_const(b)
    assert b.ndim == 1
    a = cp.reshape(a, (a.size, 1))
    b = cp.reshape(b, (1, b.size))
    expr = a @ b
    return expr

class Ergodic_Planner:
    def __init__(self, nodes, edges):
        self.num_regions = len(nodes)
        self.edges = edges
        
    def UBFMMC(self, weight, transform=True):
        weight=weight/sum(weight)
        n=len(weight)
        P= cp.Variable((n,n))


        sigma=weight
        q=np.sqrt(sigma)
        Q=np.diag(q)
        constrains=[P@sigma==sigma,
                    cp.min(P)>=0,
                    np.ones(n)@P==np.ones(n)]
        
        for i in range(n):
            for j in range(n):
                if ([i,j] not in self.edges):
                    constrains.append(P[j,i]==0)

        if transform:
            prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-outer(q,q),2)),
                                constrains)

        else:
              prob = cp.Problem(cp.Minimize(cp.norm(P-outer(sigma,np.ones(n)),2)),
                                constrains)
     

        prob.solve()
        P=P.value

        P[P<0]=0
        for i in range(n):
            for j in range(n):
                if ([i,j] not in self.edges):
                    P[j,i]=0
        P=P/sum(P, 0)

        return P
    
    def get_next_region(self, weight, current_region):
        P = self.UBFMMC(weight, self.edges)
        region=np.random.choice(range(self.num_regions),p=P[:,current_region])
        return region, P
    



# if __name__ == '__main__':

