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
    def __init__(self, costmap, region_bounds, T_camera, cam_param, img_shape):
        self.costmap = costmap
        self.region_bounds = region_bounds
        self.T_camera = T_camera
        self.K = cam_param
        self.img_shape = img_shape
        
    def get_optimal_waypoint(self,region, num_candidates, region_cloud, entropies):
        candidates = self.get_waypoints( region, num_candidates)    
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
    def get_waypoints(self, region, n):
        bound = self.region_bounds[region]
        coords = []
        while len(coords)< n:
            coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 3 )
            # coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 5)[:2]
            # centroid = 1/2*(np.array(bound["min_bound"])+np.array(bound["max_bound"]))[:2]
            cost = self.get_cost(coord_rand)
            if cost<127:
                coords.append(coord_rand)
        # cost = self.get_cost(coord_rand)
        angles = np.random.uniform(0, 2*np.pi, size = len(coords))
        coords = np.asarray(coords)
        coords[:,2] = angles
        return coords
    
    def get_index(self, coord):
        idx = (coord[0:2]-self.costmap["bounds"]["min"][0:2])/self.costmap["resolution"]
        idx = np.array([round(idx[0]), round(idx[1])])
        return idx
        
    def get_cost(self, coord):
        idx = self.get_index(coord)
        x_shape, y_shape = self.costmap["costmap"].shape
        if (idx<[0,0]).any() or (idx>=[x_shape, y_shape]).any():
            return 255
        # test = self.costmap["costmap"].copy().T
        # test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR) 
        # test = cv2.circle(test, (ind[0], ind[1]), 2, [255,0,0], -1)
        # plt.imshow(test, origin="lower")
        cost = self.costmap["costmap"][idx[0], idx[1]]
        return cost
    

if __name__ == '__main__':
    with open('tests/ref_cloud_detected.pickle', 'rb') as handle:
        ref_cloud = pickle.load(handle)
     
    with open("tests/region_bounds.yaml") as stream:
        try:
            region_bounds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)    
            
    with open('costmap.pickle', 'rb') as handle:
        costmap = pickle.load(handle)      
        
    planner = Waypoint_Planner(costmap)
    region_bounds[0]