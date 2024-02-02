# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:20:29 2024

@author: hibad
"""

from numba import cuda
import numpy as np
import time
def get_cloud_covariance(depth, Q, T):
     n, m = depth.shape
    
     J=[T@np.array([[depth[i,j],0,i],
                 [0,depth[i,j],j],
                 [0,0,1]]) for i in range(n) for j in range(m)]
 
     cov=np.asarray([j@Q[0:3,0:3]@j.T for j in J])
     return cov

TPB=32
@cuda.jit()
def cloud_cov_kernel(d_out, d_depth, d_Q, d_T):
    i,j = cuda.grid(2)
    nx, ny=d_depth.shape

    if i<nx and j<ny:
        n=int(j+i*ny)
        d = d_depth[i,j]
        J00 = d*d_T[0,0]
        J01 = d*d_T[0,1]
        J02 = d_T[0,2] + i*d_T[0,0] + j*d_T[0,1]
        J10 = d*d_T[1,0]
        J11 = d*d_T[1,1]
        J12 = d_T[1,2] + i*d_T[1,0] + j*d_T[1,1]
        J20 = d*d_T[2,0]
        J21 =  d*d_T[2,1]
        J22 = d_T[2,2] + i*d_T[2,0] + j*d_T[2,1]
    
        d_out[n,0,0] = d_Q[0,0]*J00**2 + d_Q[1,1]*J01**2 + d_Q[2,2]*J02**2
        d_out[n,0,1] = d_Q[0,0]*J00*J10 + d_Q[1,1]*J01*J11 + d_Q[2,2]*J02*J12
        d_out[n,0,2] = d_Q[0,0]*J00*J20 + d_Q[1,1]*J01*J21 + d_Q[2,2]*J02*J22
        
        d_out[n,1,0] = d_out[n,0,1] 
        d_out[n,1,1] = d_Q[0,0]*J10**2 + d_Q[1,1]*J11**2 + d_Q[2,2]*J12**2
        d_out[n,1,2] = d_Q[0,0]*J10*J20 + d_Q[1,1]*J11*J21 + d_Q[2,2]*J12*J22
        
        d_out[n,2,0] = d_out[n,0,2]
        d_out[n,2,1] = d_out[n,1,2]
        d_out[n,2,2] = d_Q[0,0]*J20**2 + d_Q[1,1]*J21**2 + d_Q[2,2]*J22**2


def get_cloud_covariance_par(depth, Q, T):
    nx, ny=depth.shape
    d_depth=cuda.to_device(depth)
    d_Q=cuda.to_device(Q)
    d_T=cuda.to_device(T)
    d_out=cuda.device_array((nx*ny, 3, 3),dtype=(np.float64))
    thread=(TPB, TPB)
    blocks=((nx+TPB-1)//TPB,(ny+TPB-1)//TPB)
    cloud_cov_kernel[blocks, thread](d_out, d_depth,d_Q, d_T)
    cov=d_out.copy_to_host()
    return cov


depth=np.random.rand(720,1280)
Q=np.eye(3)
T=np.eye(3)

t=time.time()
cov=get_cloud_covariance_par(depth, Q, T)
print("parallel:",time.time()-t)
t=time.time()
cov2=get_cloud_covariance(depth, Q, T)
print("direct:",time.time()-t)

print("difference: ", np.max(cov-cov2))