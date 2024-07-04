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
import colorsys as cs
from scipy.stats import bernoulli 
from ergodic_planner import Ergodic_Planner
import cv2 
import yaml 

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
with open("region_bounds.yaml") as stream:
    try:
        region_bounds = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
detector = Anomaly_Detector(mesh, box,region_bounds,0.02)

p=[]
for node in graph.pose_nodes.values():
    if not node.local_map == None:
        pc, ref = detector.detect(node, graph.feature_nodes)
        p.append(pc)
# pc, ref = detector.detect(graph.pose_nodes[0], graph.feature_nodes)
# p.append(pc)

# o3d.visualization.draw_geometries(p+[ref])
o3d.visualization.draw_geometries([ref])
# o3d.visualization.draw_geometries(frames+p)
#%%
hue = 0      
ref  = deepcopy(detector.reference)    
ref.paint_uniform_color([0,0,0])    
color = np.asarray(ref.colors)
n_region = len(detector.region_idx)
for idx in detector.region_idx:     
    r,g,b =  cs.hsv_to_rgb(hue, 1, 1)
    color[idx] =  np.array([r,g,b])
    hue += 1/n_region
    
box = ref.get_axis_aligned_bounding_box()
box.max_bound = [box.max_bound[0], box.max_bound[1], 0.7]
ref =  ref.crop(box) 
o3d.visualization.draw_geometries([ref])

entropy = np.zeros(n_region)
for i, idx in enumerate(detector.region_idx):
    p = detector.p_anomaly[idx]
    h = bernoulli.entropy(p)
    #entropy[i] = np.quantile(h,0.5)
    entropy[i] = np.mean(h)
    
 #%%
region_planner = Ergodic_Planner(None, None)   
region, P = region_planner.get_next_region(entropy, 0)

#%%
from waypoint_placement import Waypoint_Planner
region = 1

with open('../costmap.pickle', 'rb') as handle:
    costmap = pickle.load(handle)   
waypoint_planner = Waypoint_Planner(costmap, detector.region_bounds)

candidates = waypoint_planner.get_waypoint(None, region, 50)    
costmap = waypoint_planner.costmap["costmap"].copy().T
costmap=costmap.astype(np.uint8)
costmap = cv2.cvtColor(costmap, cv2.COLOR_GRAY2BGR) 
idxs = [waypoint_planner.get_index(x) for x in candidates]

import colorsys
T_camera = np.eye(4)
T_camera[0:3,3]= [0.077, -0.000, 0.218]
T_camera[0:3,0:3] =  [[0.0019938, -0.1555174, 0.9878311],
                      [-0.999998, -0.000156, 0.0019938],
                      [-0.000156, -0.9878331, -0.1555174]]
K = np.array([[872.2853801540007, 0.0, 604.5],
             [0.0, 872.2853801540007, 360.5],
             [ 0.0, 0.0, 1.0]])
w, h = 1208, 720

t = time.time()
reward=np.zeros(len(candidates))
# region_cloud = detector.reference.select_by_index(detector.region_idx[region])
entropies, region_cloud = detector.get_region_entropy(region)
hue = (entropies-np.min(entropies))/(np.max(entropies)-np.min(entropies))
rgb = [colorsys.hsv_to_rgb(x, 1, 1) for x in hue]
region_cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb))
tank_cloud = deepcopy((detector.reference)).paint_uniform_color([0,0,0])
frames = []
candidate_idx=[]
for i,candidate in enumerate(candidates):
    T_robot = np.eye(4)
    T_robot[0:2,3] = candidate[0:2]
    T_robot[0:2,0:2] = [[np.cos(candidate[2]), -np.sin(candidate[2])],
                        [np.sin(candidate[2]), np.cos(candidate[2])]]
    T = T_robot@T_camera
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    frame.transform(T)
    frames.append(frame)
    cloud = deepcopy(region_cloud).transform(np.linalg.inv(T))
    points = np.asarray(cloud.points)
    pixel = (K@points.T).T
    pixel_x = pixel[:,0]/pixel[:,2]
    pixel_y = pixel[:,1]/pixel[:,2]
    idx = np.where((points[:,2]<1) & (points[:,2]>0.05) & 
                    (pixel_x>=0) & (pixel_x<w) &
                    (pixel_y>=0) & (pixel_y<h))[0]
    if len(idx) == 0:
        reward[i] = 0
        global_idx=[]
    else:
        points = points[idx]
        global_idx = detector.region_idx[region][idx]
        # entropy = bernoulli.entropy(detector.p_anomaly[global_idx])
        entropy = entropies[idx]
        reward[i] =  np.sum(entropy)
    candidate_idx.append(global_idx)
    
for i, idx in enumerate(idxs):
    color = reward[i]/max(reward)*255
    costmap = cv2.circle(costmap, (idx[0], idx[1]), 2, [color,0,0], -1)
plt.imshow(costmap, origin="lower")
        
print(time.time()-t)
idx = np.argmax(reward)
colors = np.asarray(tank_cloud.colors)
colors[np.asarray(candidate_idx[idx])] = [255,0,0]
tank_cloud.colors=o3d.utility.Vector3dVector(colors)
tank_cloud = tank_cloud.crop(detector.bounding_box)
o3d.visualization.draw_geometries([tank_cloud, frames[idx]])

# p=o3d.geometry.PointCloud()
# p.points=o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([test, o3d.geometry.TriangleMesh.create_coordinate_frame(1)]+frames)

