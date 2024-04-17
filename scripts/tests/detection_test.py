#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:45:28 2024

@author: hibad
"""
# import sys
# sys.path.append('../')
import numpy as np
from anomaly_detector import Anomaly_Detector
import pickle 
import matplotlib.pyplot as plt 
import time 
from copy import deepcopy
from hierarchical_SLAM_SE3 import Graph_SLAM
from Lie import SE3
from numpy import sin, cos
import open3d as o3d

np.set_printoptions(precision=2)
with open('graph.pickle', 'rb') as f:
    graph = pickle.load(f)


mesh = o3d.io.read_triangle_mesh("ballast.STL")
box = mesh.get_axis_aligned_bounding_box()
bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
box.max_bound = bound

detector = Anomaly_Detector(mesh, box,0.03)
p=[]
# for node in graph.pose_nodes.values():
#     if not node.local_map == None:
#         pc, ref = detector.detect(node)
#         p.append(pc)
pc, ref = detector.detect(graph.pose_nodes[0])
p.append(pc)

# o3d.visualization.draw_geometries(p+[ref])
# o3d.visualization.draw_geometries([ref])
o3d.visualization.draw_geometries(p)
