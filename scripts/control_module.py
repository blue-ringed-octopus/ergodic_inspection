#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:12:36 2024

@author: hibad
"""

from ergodic_planner import Ergodic_Planner
from waypoint_placement import Waypoint_Planner
import pickle
import open3d as o3d
import matplotlib.pyplot as plt 
import cv2
from map_manager import Map_Manager
import numpy as np
from anomaly_detector import Anomaly_Detector
# import rospy
# import rospkg

# rospack=rospkg.RosPack()
# path = rospack.get_path("ergodic_inspection")

# if __name__ == '__main__':
from scipy.stats import bernoulli 

# rospy.wait_for_service('get_reference_cloud_region')
with open('tests/graph.pickle', 'rb') as f:
    graph = pickle.load(f)
frames=[]


for item in graph.feature_nodes.values():
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    frame.transform(item.M)
    frames.append(frame)
mesh = o3d.io.read_triangle_mesh("../resources/ballast.STL")
box = mesh.get_axis_aligned_bounding_box()
bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
box.max_bound = bound
   


map_manager = Map_Manager("../")
T_camera = np.eye(4)
T_camera[0:3,3]= [0.077, -0.000, 0.218]
T_camera[0:3,0:3] =  [[0.0019938, -0.1555174, 0.9878311],
                      [-0.999998, -0.000156, 0.0019938],
                      [-0.000156, -0.9878331, -0.1555174]]
K = np.array([[872.2853801540007, 0.0, 604.5],
             [0.0, 872.2853801540007, 360.5],
             [ 0.0, 0.0, 1.0]])
w, h = 1208, 720

detector = Anomaly_Detector(map_manager.reference, box,0.02)
for node in graph.pose_nodes.values():
    if not node.local_map == None:
        pc, ref = detector.detect(node, graph.feature_nodes)
        
#%%  
import colorsys
map_manager.set_entropy(detector.p_anomaly)      
entropies = np.zeros(len(map_manager.region_idx))
for i, idx in enumerate(map_manager.region_idx):
    entropy = map_manager.h[idx]
    entropies[i] = np.mean(entropy)

region_planner = Ergodic_Planner(map_manager.hierarchical_graph.levels[1])
region, P = region_planner.get_next_region(entropies, 0)
waypoint_planner = Waypoint_Planner(map_manager.costmap, map_manager.region_bounds, T_camera, K, (w,h))
entropies, region_cloud = map_manager.get_region_entropy(region)

waypoint = waypoint_planner.get_optimal_waypoint(region, 50, region_cloud, entropies)
T_robot = np.eye(4)
T_robot[0:2,3] = waypoint[0:2]
T_robot[0:2,0:2] = [[np.cos(waypoint[2]), -np.sin(waypoint[2])],
                   [np.sin(waypoint[2]), np.cos(waypoint[2])]]

hue = (entropies-np.min(entropies))/(np.max(entropies)-np.min(entropies))
rgb = [colorsys.hsv_to_rgb(x, 1, 1) for x in hue]
region_cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb))

robot_frame  = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
robot_frame = robot_frame.transform(T_robot)
o3d.visualization.draw_geometries([region_cloud, robot_frame])
    