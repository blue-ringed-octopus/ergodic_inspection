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
    def __init__(self, strategy, costmap, T_camera, cam_param, img_shape):
        self.strategy = strategy
        self.costmap = costmap
        self.T_camera = T_camera
        self.K = cam_param
        self.img_shape = img_shape
    
    def camera_projection(self, pose, point_cloud):
        w, h = self.img_shape
        T_robot = np.eye(4)
        T_robot[0:2,3] = pose[0:2]
        T_robot[0:2,0:2] = [[np.cos(pose[2]), -np.sin(pose[2])],
                            [np.sin(pose[2]), np.cos(pose[2])]]
        T = T_robot@self.T_camera
        cloud = deepcopy(point_cloud).transform(np.linalg.inv(T))
        points = np.asarray(cloud.points)
        pixel = (self.K@points.T).T
        pixel_x = pixel[:,0]/pixel[:,2]
        pixel_y = pixel[:,1]/pixel[:,2]
        idx = np.where((points[:,2]<2) & (points[:,2]>0.05) & 
                        (pixel_x>=0) & (pixel_x<w) &
                        (pixel_y>=0) & (pixel_y<h))[0]

        return idx
        
    
    def get_optimal_waypoint(self,num_candidates, region_cloud, entropies):
        bound={}
        bound["min_bound"] = region_cloud.get_min_bound()
        bound["max_bound"] = region_cloud.get_max_bound()
        candidates = self.sample_waypoints( bound, num_candidates)    
        reward=np.zeros(len(candidates))

        for i,candidate in enumerate(candidates):
            idx = self.camera_projection(candidate, region_cloud)
            if len(idx) == 0:
                reward[i] = 0
            else:
                # entropy = bernoulli.entropy(detector.p_anomaly[global_idx])
                entropy = entropies[idx]
                reward[i] =  np.sum(entropy)
        w = reward/np.sum(reward)
        if self.strategy == "ergodic":
            idx = np.random.choice(range(len(w)), p = w)
        elif self.strategy == "random":
            idx = np.random.choice(range(len(w)))
        elif self.strategy == "greedy":
            idx = np.argmax(reward)
        waypoint = candidates[idx]
        
        return np.array(waypoint)
    
    def sample_waypoints(self, bound, n):
        coords = []
        while len(coords)< n:
            coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 3 )
            # coord_rand= np.random.uniform(bound["min_bound"], bound["max_bound"], size = 5)[:2]
            # centroid = 1/2*(np.array(bound["min_bound"])+np.array(bound["max_bound"]))[:2]
            cost = self.get_cost(coord_rand)
            if cost<20:
                print(cost)
                coords.append(coord_rand)
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

        cost = self.costmap["costmap"][idx[0], idx[1]]
        return cost
    def plot_waypoints(self, waypoints):
        im = np.round(self.costmap["costmap"].copy()*255/100).astype(np.uint8)
        im = cv2.cvtColor(im.T, cv2.COLOR_GRAY2RGB)
        for waypoint in waypoints:
            idx = self.get_index(waypoint[0:2])
            idx2 = self.get_index(waypoint[0:2]+ [0.5*np.cos(waypoint[2]),0.5*np.sin(waypoint[2]) ])
            im = cv2.arrowedLine(im, (int(idx[0]), int(idx[1])), (int(idx2[0]),int(idx2[1])),
                                             (255,0,0),1)  
        return im
if __name__ == '__main__':
    from map_manager import Map_Manager
    import open3d as o3d
    from scipy.stats import bernoulli 
    import colorsys
    
    manager = Map_Manager("../")
    with open('tests/detections.pickle', 'rb') as f:
        dat = pickle.load(f)
        
    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(dat["cloud"])
    manager.process_reference(pc)
    manager.set_entropy(dat["p"][-1], np.array(range(len(dat["p"][-1]))))
     
    T_camera = np.eye(4)
    T_camera[0:3,3]= [0.077, -0.000, 0.218]
    T_camera[0:3,0:3] =  [[0.0019938, -0.1555174, 0.9878311],
                          [-0.999998, -0.000156, 0.0019938],
                          [-0.000156, -0.9878331, -0.1555174]]
    
    K = np.array([[872.2853801540007, 0.0, 604.5],
                 [0.0, 872.2853801540007, 360.5],
                 [ 0.0, 0.0, 1.0]])

    w, h = 1208, 720
    region = 5
    entropy, cloud = manager.get_region_entropy(region)  
    v = 1 - entropy/bernoulli.entropy(0.5)
    rgb = [colorsys.hsv_to_rgb(0, 0, v[i]) for i in range(len(entropy))]
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb))
    
    planner = Waypoint_Planner("ergodic", manager.costmap,T_camera, K, (w,h))    
    waypoint = planner.get_optimal_waypoint(1000, cloud, entropy)
    idx = planner.camera_projection(waypoint, cloud)
    colors = np.asarray(cloud.colors)
    colors[idx] = [255,0,0]
    cloud.colors = o3d.utility.Vector3dVector(colors)
    T_robot = np.eye(4)
    T_robot[0:2,3] = waypoint[0:2]
    T_robot[0:2,0:2] = [[np.cos(waypoint[2]), -np.sin(waypoint[2])],
                       [np.sin(waypoint[2]), np.cos(waypoint[2])]]
    robot_frame  = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    robot_frame = robot_frame.transform(T_robot)
    o3d.visualization.draw_geometries([cloud, robot_frame])
    
    im = planner.plot_waypoints([waypoint])
    plt.imshow(im, origin="lower")
    
