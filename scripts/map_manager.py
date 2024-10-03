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
        with open(path+'costmap.pickle', 'rb') as handle:
            costmap = pickle.load(handle)  
        self.costmap = costmap
        self.build_region_graph()
        self.build_reference_pointcloud()
        self.p = np.zeros(self.num_points)
        self.h = np.zeros(self.num_points)
        self.set_entropy(np.ones(self.num_points)*0.5,np.array(range(self.num_points)))
        
    def build_reference_pointcloud(self):
        mesh = o3d.io.read_triangle_mesh(self.path + "ballast.STL")
        self.mesh = mesh
        num_points = 100000
        self.num_points = num_points

        pc = mesh.sample_points_uniformly(
            number_of_points=num_points, use_triangle_normal=True)
        self.process_reference(pc)
        
        
    def process_reference(self, pc):  
        pc.paint_uniform_color([0,0,0])
        box = pc.get_axis_aligned_bounding_box()
        min_bound = [box.min_bound[0],box.min_bound[1], 0.02 ]
        max_bound = [box.max_bound[0],box.max_bound[1], 0.5 ]
        box.min_bound = min_bound
        box.max_bound = max_bound
        self.bound = box
        self.reference = deepcopy(pc)
        self.ref_normal = np.asarray(pc.normals)
        self.ref_points = np.asarray(pc.points)
        self.ref_tree = KDTree(self.ref_points)
        self.partition()

    def partition(self):
        region_bounds=self.region_bounds
        region_idx={}
        ref  = deepcopy(self.reference)    
        for id_, bound in region_bounds.items():     
            box = ref.get_axis_aligned_bounding_box()
            box.max_bound = bound["max_bound"]
            box.min_bound = bound["min_bound"]
            region = ref.crop(box)
            _, idx = self.ref_tree.query(region.points)
            region_idx[id_] = idx
            
        self.region_idx = region_idx
        
    def build_region_graph(self):
        with open(self.path+"region_bounds.yaml") as stream:
            try:
                region_bounds = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)  
        self.region_bounds = region_bounds
        mask = cv2.threshold(self.costmap["costmap"].copy(), 50, 100,  cv2.THRESH_BINARY)[1]
        root_node = Hierarchical_Graph.Node(0,[0,0], 0)
        root_grid = mask.astype(np.int32)
        root_grid[root_grid==100] = -1
        root_graph = Graph({0:root_node} , 0 ,root_grid)
        h_graph = Hierarchical_Graph(root_graph)
        
        region_map = np.zeros(root_grid.shape)
        for region_id, bounds in region_bounds.items():
            min_idx = self.get_index(bounds["min_bound"][0:2])
            max_idx = self.get_index(bounds["max_bound"][0:2])
            region_map[min_idx[0]:max_idx[0],min_idx[1]:max_idx[1]] = region_id
        idx_map = region_map.copy()
        
        idx_map[root_grid==-1] = -1 

        region_nodes={}
        region_idx=[]
        for i, x in enumerate(region_bounds):
           idx= np.array(np.where(idx_map==x)).T
           region_idx.append(idx)
           region_nodes[i] = Hierarchical_Graph.Node(i,np.mean(idx,0), 1)
           region_nodes[i].add_parent(root_node)
        
        region_graph =  Graph(region_nodes , 1 ,idx_map)
        h_graph.levels[1] = region_graph
        
        stencil = [[i,j] for i in [-1,0,1] for j in [-1,0,1] if not(i==0 and j==0)]
        
        grid_nodes, grid_ids = h_graph.grid2graph(stencil, 2)
        grid_graph = Graph(grid_nodes , 2 ,grid_ids)
        h_graph.levels[2] = grid_graph
        self.hierarchical_graph = h_graph
        self.region_map = region_map
        
    def get_index(self, coord):
        idx = (coord[0:2]-self.costmap["bounds"]["min"][0:2])/self.costmap["resolution"]
        idx = np.array([round(idx[0]), round(idx[1])])
        return idx
    
    def get_region_graph_img(self):
        return self.hierarchical_graph.levels[1].plot_graph()
    
    def set_entropy(self, p, idx):
        self.p[idx] = p
        self.h[idx] = bernoulli.entropy(p)
        
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
        v = 1 - self.h.copy()/bernoulli.entropy(0.5)
        h = (1 - self.p.copy())*1/3
        rgb = [colorsys.hsv_to_rgb(h[i], 1, v[i]) for i in range(self.num_points)]
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb))
        return cloud.crop(self.bound)
    
    def get_entropy(self):
        entropy=np.zeros(len(self.region_idx))
        for i, idx in self.region_idx.items():
            h = self.h[idx]
            # entropy[i] = np.quantile(h,0.25)
            # entropy[i] = np.quantile(h,0.75)
            # entropy[i] = np.quantile(h,0.5)
            # entropy[i] = np.mean(h)
            entropy[i] = np.sum(h)
        return entropy
    
    def get_graph(self, level):
        ids = list(self.hierarchical_graph.levels[level].nodes.keys())
        edges = self.hierarchical_graph.get_edges(level)
        h = self.get_entropy()
        # location = np.array([ node.coord for node in self.hierarchical_graph.levels[level].nodes.values()])
        return ids,  edges, h
    
    def coord_to_region(self, coord, level):
        idx = self.get_index(coord)
        # region = self.hierarchical_graph.levels[level].id_map[idx[0], idx[1]]
        region = self.region_map[idx[0], idx[1]]
        return int(region)
    
    def draw_graph_entropy(self):
        graph = manager.hierarchical_graph.levels[1]
        h = self.get_entropy()
        h = h/np.max(h) 
        n = len(ids)
        v = 1 - h.copy()/bernoulli.entropy(0.5)
        region_color = [(colorsys.hsv_to_rgb(0, 0, v[i])) for i in range(n)]
        img = graph.id_map.copy()
        img = img.astype(np.float32)
        img[img>=0] = 255
        img[img==-1] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(n):
            img[graph.id_map == i, :] = region_color[i]
        h, w, _ = img.shape
        scale = 500
        img = cv2.resize(img, (int(scale), int(h/w*scale)),interpolation = cv2.INTER_NEAREST)
        for i, node in enumerate(graph.nodes.values()):
            textsize = cv2.getTextSize(str(node.id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            pos = (int(scale/w*node.coord[1]), int(scale/w*node.coord[0]))
            img = cv2.circle(img, pos ,20, 0, -1)
            pos = (int(scale/w*node.coord[1]-textsize[1]/2), int(scale/w*node.coord[0]+textsize[0]/2))
            img = cv2.putText(img, str(node.id), pos, cv2.FONT_HERSHEY_SIMPLEX , 1, region_color[i] , 2)
            for neighbor in node.neighbor_nodes.values():
                pos2 = (int(scale/w*neighbor.coord[1]), int(scale/w*neighbor.coord[0]))
                img = cv2.arrowedLine(img, pos, pos2, 0)
        return img
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    manager = Map_Manager("../resources/")
    with open('tests/detections.pickle', 'rb') as f:
        dat = pickle.load(f)
        
    pc=o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(dat["cloud"])
    manager.process_reference(pc)
    manager.set_entropy(dat["p"][-1], np.array(range(len(dat["p"][-1]))))
    test = manager.visualize_entropy()
    # o3d.visualization.draw_geometries([test])
    print(manager.get_entropy())
    img = manager.get_region_graph_img()
    plt.figure(dpi=1200)   
    plt.imshow(img)    
    ids, edges , h = manager.get_graph(1)
    region = manager.coord_to_region([2.57,-0.75], 1)
    img2 = manager.draw_graph_entropy()
    plt.figure()   
    plt.imshow(img2)    


   