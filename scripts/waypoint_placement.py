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

class Waypoint_Planner:
    def __init__(self, costmap, region_bounds):
        self.costmap = costmap
        self.region_bounds = region_bounds
        
    def get_waypoint(self, pointcloud, region, n):
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