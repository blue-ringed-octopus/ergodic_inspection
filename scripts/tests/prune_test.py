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
np.set_printoptions(precision=2)

with open('graph.pickle', 'rb') as handle:
    graph = pickle.load(handle)

graph_prune = deepcopy(graph)
graph_prune.prune()
prior = graph_prune.prior_factor

optimizer = Graph_SLAM.Back_end()
optimizer.optimize(graph_prune)