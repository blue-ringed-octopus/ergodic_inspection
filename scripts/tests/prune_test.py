# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:14:24 2024

@author: hibad
"""

# import sys
# sys.path.append('../')

import numpy as np

import pickle 
import matplotlib.pyplot as plt 
import time 
from copy import deepcopy
from hierarchical_SLAM_SE3 import Graph_SLAM
from Lie import SE3
from numpy import sin, cos
np.set_printoptions(precision=2)
#%%
with open('graph.pickle', 'rb') as f:
    graph = pickle.load(f)

#%%
graph_prune = deepcopy(graph)

optimizer = Graph_SLAM.Back_end()
optimizer.optimize(graph_prune)
graph_prune.prune(1)
optimizer.optimize(graph_prune)

#%%
plt.figure(1)
prior = graph_prune.prior_factor
plt.plot(0, 0, "s", color=(0,0,0), markersize = 10)

for child in prior.children:
    i = prior.idx_map["pose"][child.id]
    M2 = SE3.Exp(prior.z[6*i:6*i+6])
    plt.plot(M2[0,3], M2[1,3], "o", color=(0.5,0.5,0.5, 0.5), markersize = 20)
    plt.plot((0, M2[0,3]), (0, M2[1,3]), "--", color=(0.5,0.5,0.5))

for feature in prior.feature_nodes:
    i = prior.idx_map["features"][feature.id]
    M2 = SE3.Exp(prior.z[6*i:6*i+6])
    plt.plot(M2[0,3], M2[1,3], "*", color=(0.5,0.5,0.5, 0.5), markersize = 30)
    plt.plot((0, M2[0,3]), (0, M2[1,3]), "--", color=(0.5,0.5,0.5))
 
    
for factor in graph_prune.factors.values():
      M1 = factor.parent.M
      M2 = factor.children[0].M
      centroid = [(M1[0,3]+M2[0,3])/2, (M1[1,3]+M2[1,3])/2]
      plt.plot((M1[0,3], M2[0,3]), (M1[1,3], M2[1,3]), "--", color=(0.5,0.5,0.5))
    # plt.plot(centroid[0], centroid[1] , "s", color='k')
      M2 = M1@SE3.Exp(factor.z[0:6])
      plt.plot(M2[0,3], M2[1,3], "o", color=(0.5,0.5,0.5, 0.5), markersize = 20)

      for feature in factor.feature_nodes:
          i=6*factor.idx_map["features"][feature.id]
          M2 = M1@SE3.Exp(factor.z[i:i+6])
          plt.plot(M2[0,3], M2[1,3], "*", color=(0.5,0.5,0.5, 0.5), markersize = 30)
          plt.plot((M1[0,3], M2[0,3]), (M1[1,3], M2[1,3]), "--", color=(0.5,0.5,0.5))

          M2 = feature.M
          plt.plot((centroid[0], M2[0,3]), (centroid[1], M2[1,3]), "--", color=(0.5,0.5,0.5))
            
for node in graph_prune.pose_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    # print(M)
    plt.plot(M[0,3], M[1,3], "o", color="k")
    plt.arrow(M[0,3], M[1,3], 0.5*cos(mu[5]), 0.5*sin(mu[5]))
    
for node in graph_prune.feature_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    # print(M)
    plt.text(M[0,3], M[1,3], node.id)
    plt.plot(M[0,3], M[1,3], "*", markersize=15)
    plt.arrow(M[0,3], M[1,3], 0.5*cos(mu[5]), 0.5*sin(mu[5]))
    


plt.axis('scaled')
# plt.xlim([-6, 0])
# plt.ylim([-1,4])
plt.title("optmized")