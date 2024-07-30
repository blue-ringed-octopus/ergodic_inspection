# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:36:09 2024

@author: hibad
"""
import numpy as np
import pickle
import yaml 
import cv2 
import matplotlib.pyplot as plt
from copy import deepcopy
class Waypoint_Planner:
    def __init__(self, costmap, T_camera, cam_param, img_shape):
        self.costmap = costmap
        self.T_camera = T_camera
        self.K = cam_param
        self.img_shape = img_shape
        
    def get_optimal_waypoint(self,num_candidates, region_cloud, entropies):
        bound={}
        bound["min_bound"] = region_cloud.get_min_bound()
        bound["max_bound"] = region_cloud.get_max_bound()
        candidates = self.get_waypoints( bound, num_candidates)    
        reward=np.zeros(len(candidates))
        w, h = self.img_shape
        for i,candidate in enumerate(candidates):
            T_robot = np.eye(4)
            T_robot[0:2,3] = candidate[0:2]
            T_robot[0:2,0:2] = [[np.cos(candidate[2]), -np.sin(candidate[2])],
                                [np.sin(candidate[2]), np.cos(candidate[2])]]
            T = T_robot@self.T_camera
            cloud = deepcopy(region_cloud).transform(np.linalg.inv(T))
            points = np.asarray(cloud.points)
            pixel = (self.K@points.T).T
            pixel_x = pixel[:,0]/pixel[:,2]
            pixel_y = pixel[:,1]/pixel[:,2]
            idx = np.where((points[:,2]<1) & (points[:,2]>0.05) & 
                            (pixel_x>=0) & (pixel_x<w) &
                            (pixel_y>=0) & (pixel_y<h))[0]
            if len(idx) == 0:
                reward[i] = 0
            else:
                points = points[idx]
                # entropy = bernoulli.entropy(detector.p_anomaly[global_idx])
                entropy = entropies[idx]
                reward[i] =  np.sum(entropy)
        idx = np.argmax(reward)
        waypoint = candidates[idx]
        
        return waypoint
    def get_waypoints(self, bound, n):
        coords = []
        while len(coords)< n:
            coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 3 )
            # coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 5)[:2]
            # centroid = 1/2*(np.array(bound["min_bound"])+np.array(bound["max_bound"]))[:2]
            cost = self.get_cost(coord_rand)
            if cost>50:
                coords.append(coord_rand)
        # cost = self.get_cost(coord_rand)
        angles = np.random.uniform(0, 2*np.pi, size = len(coords))
        coords = np.asarray(coords)
        coords[:,2] = angles
        return coords
    
    def get_index(self, coord):
        idx = (coord[0:2]-self.costmap["origin"][0:2])/self.costmap["resolution"]
        idx = np.array([round(idx[0]), round(idx[1])])
        return idx
        
    def get_cost(self, coord):
        idx = self.get_index(coord)
        x_shape, y_shape = self.costmap["costmap"].shape
        if (idx<[0,0]).any() or (idx>=[x_shape, y_shape]).any():
            return 100
        # test = self.costmap["costmap"].copy().T
        # test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR) 
        # test = cv2.circle(test, (ind[0], ind[1]), 2, [255,0,0], -1)
        # plt.imshow(test, origin="lower")
        cost = self.costmap["costmap"][idx[0], idx[1]]
        return cost
    

if __name__ == '__main__':
    from map_manager import Map_Manager
    
    manager = Map_Manager("../")
    with open('tests/ref_cloud_detected.pickle', 'rb') as handle:
        ref_cloud = pickle.load(handle)
     
    T_camera = np.eye(4)
    T_camera[0:3,3]= [0.077, -0.000, 0.218]
    T_camera[0:3,0:3] =  [[0.0019938, -0.1555174, 0.9878311],
                          [-0.999998, -0.000156, 0.0019938],
                          [-0.000156, -0.9878331, -0.1555174]]
    K = np.array([[872.2853801540007, 0.0, 604.5],
                 [0.0, 872.2853801540007, 360.5],
                 [ 0.0, 0.0, 1.0]])
    w, h = 1208, 720
    entropy, cloud = manager.get_region_entropy(1)  
    planner = Waypoint_Planner(manager.costmap,T_camera, K, (w,h))    
    waypoint = planner.get_optimal_waypoint(10, cloud, entropy)
