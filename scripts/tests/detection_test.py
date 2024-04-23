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
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=2)
with open('graph.pickle', 'rb') as f:
    graph = pickle.load(f)
frames=[]

feature_nodes_prior = {}
M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, -1.57]).as_matrix()
M[0:3,3] = [-2.516, 1.326, 0.227]
feature_nodes_prior[14] =  Graph_SLAM.Front_end.Node(14,M.copy(),"feature")
M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 0]).as_matrix()
M[0:3,3] =  [-4.054, 0.344, 0.1278]
feature_nodes_prior[0] =  Graph_SLAM.Front_end.Node(0,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 0]).as_matrix()
M[0:3,3] =  [-5.051, 1.3464, 0.1995]
feature_nodes_prior[1] =  Graph_SLAM.Front_end.Node(1,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 0]).as_matrix()
M[0:3,3] =  [-5.05, 0.422, 0.227]
feature_nodes_prior[13] =  Graph_SLAM.Front_end.Node(13,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 0]).as_matrix()
M[0:3,3] =  [-5.051, 2.25, 0.227 ]
feature_nodes_prior[17] =  Graph_SLAM.Front_end.Node(17,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 3.14]).as_matrix()
M[0:3,3] =  [-4.0748, 2.769, 0.1068]
feature_nodes_prior[2] =  Graph_SLAM.Front_end.Node(2,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 0]).as_matrix()
M[0:3,3] =  [-5.0505, 3.16, 0.227]
feature_nodes_prior[19] =  Graph_SLAM.Front_end.Node(19,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [0, 0, 3.14]).as_matrix()
M[0:3,3] =  [-2.381, 2.7281, 0.227]
feature_nodes_prior[18] =  Graph_SLAM.Front_end.Node(18,M.copy(),"feature")



M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [ 0, 0, 3.14]).as_matrix()
M[0:3,3] =  [-0.574,3.174, 0.227]
feature_nodes_prior[15] =  Graph_SLAM.Front_end.Node(15,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [ 0, 0, 3.14]).as_matrix()
M[0:3,3] =  [-0.360, 2.25, 0.227]
feature_nodes_prior[20] =  Graph_SLAM.Front_end.Node(20,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [ 0, 0,3.14]).as_matrix()
M[0:3,3] =  [-0.3587, 1.333, 0.146]
feature_nodes_prior[3] =  Graph_SLAM.Front_end.Node(3,M.copy(),"feature")

M = np.eye(4)
M[0:3,0:3] = R.from_euler('xyz', [ 0, 0, 3.14]).as_matrix()
M[0:3,3] =  [-0.360 , 0.4097, 0.166]
feature_nodes_prior[23] =  Graph_SLAM.Front_end.Node(23,M.copy(),"feature")




graph.feature_nodes[14] = feature_nodes_prior[14]
graph.feature_nodes[0] = feature_nodes_prior[0]
graph.feature_nodes[1] = feature_nodes_prior[1]
graph.feature_nodes[13] = feature_nodes_prior[13]
graph.feature_nodes[17] = feature_nodes_prior[17]
graph.feature_nodes[2] = feature_nodes_prior[2]
graph.feature_nodes[19] = feature_nodes_prior[19]
graph.feature_nodes[18] = feature_nodes_prior[18]
graph.feature_nodes[15] = feature_nodes_prior[15]
graph.feature_nodes[20] = feature_nodes_prior[20]
graph.feature_nodes[3] = feature_nodes_prior[3]
graph.feature_nodes[23] = feature_nodes_prior[23]

for item in graph.feature_nodes.values():
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    frame.transform(item.M)
    frames.append(frame)
mesh = o3d.io.read_triangle_mesh("ballast.STL")
box = mesh.get_axis_aligned_bounding_box()
bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
box.max_bound = bound

detector = Anomaly_Detector(mesh, box,0.01)
p=[]
for node in graph.pose_nodes.values():
    if not node.local_map == None:
        pc, ref = detector.detect(node, graph.feature_nodes)
        p.append(pc)
# pc, ref = detector.detect(graph.pose_nodes[0], graph.feature_nodes)
# p.append(pc)

# o3d.visualization.draw_geometries(p+[ref])
o3d.visualization.draw_geometries([ref])
o3d.visualization.draw_geometries(frames+p)
