#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:53:55 2024

@author: hibad
"""
import yaml
import pickle
import cv2 
import numpy as np
from hierarchical_graph import Hierarchical_Graph, Graph
from scipy.stats import bernoulli 
import open3d as o3d 
from copy import deepcopy 
from scipy.spatial import KDTree
import colorsys

class Map_Manager:
    def __init__(self, path):
        self.path = path
        with open(path+'/resources/costmap.pickle', 'rb') as handle:
            costmap = pickle.load(handle)  
        self.costmap = costmap
        self.build_region_graph()
        self.build_reference_pointcloud()
        self.partition()
        self.set_entropy(np.ones(self.num_points)*0.5)
        
    def build_reference_pointcloud(self):
        mesh = o3d.io.read_triangle_mesh(self.path + "/resources/ballast.STL")
        self.mesh = mesh
        num_points = 100000
        self.num_points = num_points
        pc = mesh.sample_points_uniformly(
            number_of_points=num_points, use_triangle_normal=True)
        pc.paint_uniform_color([0,0,0])
        self.reference = deepcopy(pc)
        self.ref_normal = np.asarray(pc.normals)
        self.ref_points = np.asarray(pc.points)
        self.ref_tree = KDTree(self.ref_points)
        box = pc.get_axis_aligned_bounding_box()
        bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
        box.max_bound = bound
        self.bound = box
        
    def partition(self):
        region_bounds=self.region_bounds
        region_idx=[]
        ref  = deepcopy(self.reference)    
        for bound in region_bounds.values():     
            box = ref.get_axis_aligned_bounding_box()
            box.max_bound = bound["max_bound"]
            box.min_bound = bound["min_bound"]
            region = ref.crop(box)
            _, idx = self.ref_tree.query(region.points)
            region_idx.append(idx)
            
        self.region_idx = region_idx
        
    def build_region_graph(self):
        with open(self.path+"/resources/region_bounds.yaml") as stream:
            try:
                region_bounds = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)  
        self.region_bounds = region_bounds
        mask = cv2.threshold(self.costmap["costmap"].copy(), 127, 255,  cv2.THRESH_BINARY)[1]
        root_node = Hierarchical_Graph.Node(0,[0,0], 0)
        root_grid = mask.astype(np.int32)
        root_grid[root_grid==255] = -1
        root_graph = Graph({0:root_node} , 0 ,root_grid)
        h_graph = Hierarchical_Graph(root_graph)
        
        region_map = np.zeros(root_grid.shape)
        for region_id, bounds in region_bounds.items():
            min_idx = self.get_index(bounds["min_bound"][0:2])
            max_idx = self.get_index(bounds["max_bound"][0:2])
            region_map[min_idx[0]:max_idx[0],min_idx[1]:max_idx[1]] = region_id
        region_map[root_grid==-1] = -1 
           
        region_nodes={}
        region_idx=[]
        for i, x in enumerate(region_bounds):
           idx= np.array(np.where(region_map==x)).T
           region_idx.append(idx)
           region_nodes[i] = Hierarchical_Graph.Node(i,np.mean(idx,0), 1)
           region_nodes[i].add_parent(root_node)
        
        region_graph =  Graph(region_nodes , 1 ,region_map)
        h_graph.levels[1] = region_graph
        
        stencil = [[i,j] for i in [-1,0,1] for j in [-1,0,1] if not(i==0 and j==0)]
        
        grid_nodes, grid_ids = h_graph.grid2graph(stencil, 2)
        grid_graph = Graph(grid_nodes , 2 ,grid_ids)
        h_graph.levels[2] = grid_graph
        self.hierarchical_graph = h_graph
        
        
    def get_index(self, coord):
        idx = (coord[0:2]-self.costmap["bounds"]["min"][0:2])/self.costmap["resolution"]
        idx = np.array([round(idx[0]), round(idx[1])])
        return idx
    
    def get_region_graph_img(self):
        return self.hierarchical_graph.levels[1].plot_graph()
    
    def set_entropy(self, p):
        self.p = p
        self.h = bernoulli.entropy(p)
        
    def get_region_entropy(self, region_id):
        if region_id==-1:
            return self.h, deepcopy(self.reference)    
        else:
            idx = self.region_idx[region_id]
            region = self.reference.select_by_index(idx)
            h = self.h[idx]
            return h, deepcopy(region) 
        
    def visualize_entropy(self):
        cloud = deepcopy(self.reference)
        v = 1 - self.h/bernoulli.entropy(0.5)
        rgb = [colorsys.hsv_to_rgb(0, 0, x) for x in v]
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb))
        return cloud.crop(self.bound)
    
    def get_entropy(self):
        entropy=np.zeros()
        for i, idx in enumerate(self.region_idx):
            h = self.h[idx]
            entropy[i] = np.quantile(h,0.5)
            # entropy[i] = np.mean(h)
        return entropy 
    
    def get_graph(self, level):
        ids = list(self.hierarchical_graph.levels[level].nodes.keys())
        edges = self.hierarchical_graph.get_edges(level)
        h = self.get_entropy()
        return ids, edges, h
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    manager = Map_Manager("../")
    img = manager.get_region_graph_img()
    plt.imshow(img)    
    ids, edges = manager.get_graph(1)
