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
from copy import deepcopy
import yaml
import colorsys as cs
from scipy.stats import bernoulli 
from ergodic_planner import Ergodic_Planner
np.set_printoptions(precision=2)
with open('graph.pickle', 'rb') as f:
    graph = pickle.load(f)
frames=[]


for item in graph.feature_nodes.values():
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    frame.transform(item.M)
    frames.append(frame)
mesh = o3d.io.read_triangle_mesh("ballast.STL")
box = mesh.get_axis_aligned_bounding_box()
bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
box.max_bound = bound

detector = Anomaly_Detector(mesh, box,0.02)
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
#%%
with open("region_bounds.yaml") as stream:
    try:
        region_bounds = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
hue = 0      
region_idx=[]
ref  = deepcopy(detector.reference)    
ref.paint_uniform_color([0,0,0])    
color = np.asarray(ref.colors)
for bound in region_bounds.values():     
    box = ref.get_axis_aligned_bounding_box()
    box.max_bound = bound["max_bound"]
    box.min_bound = bound["min_bound"]
    region = ref.crop(box)
    _, idx = detector.ref_tree.query(region.points)
    region_idx.append(idx)
    r,g,b =  cs.hsv_to_rgb(hue, 1, 1)
    color[idx] =  np.array([r,g,b])
    hue += 1/len(region_bounds)
    
box = ref.get_axis_aligned_bounding_box()
box.max_bound = [box.max_bound[0], box.max_bound[1], 0.7]
ref =  ref.crop(box) 
o3d.visualization.draw_geometries([ref])

entropy = np.zeros(len(region_bounds))
for i, idx in enumerate(region_idx):
    p = detector.p_anomaly[idx]
    h = bernoulli.entropy(p)
    #entropy[i] = np.quantile(h,0.5)
    entropy[i] = np.mean(h)
    
 #%%
planner = Ergodic_Planner(None, None)   
region, P = planner.get_next_region(entropy, 0)