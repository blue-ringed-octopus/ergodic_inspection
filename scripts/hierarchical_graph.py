# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:13:16 2024

@author: hibad
"""
import cv2
import numpy as np
import colorsys


class Graph:
    def __init__(self, nodes, level,id_map):
        self.nodes =nodes  
        self.level = level
        self.id_map = id_map
        self.edges = []
        
    def plot_graph(self):
        n = len(self.nodes)
        region_color = [(colorsys.hsv_to_rgb(i/n, 0.5, 1)) for i in range(n)]
        img = self.id_map.copy()
        img = img.astype(np.float32)
        img[img>=0] = 255
        img[img==-1] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(n):
            img[self.id_map == i, :] = region_color[i]
        h, w, _ = img.shape
        scale = 500
        img = cv2.resize(img, (int(scale), int(h/w*scale)),interpolation = cv2.INTER_NEAREST)
        for i, node in enumerate(self.nodes.values()):
            textsize = cv2.getTextSize(str(node.id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            pos = (int(scale/w*node.coord[1]), int(scale/w*node.coord[0]))
            img = cv2.circle(img, pos ,20, 0, -1)
            pos = (int(scale/w*node.coord[1]-textsize[1]/2), int(scale/w*node.coord[0]+textsize[0]/2))
            img = cv2.putText(img, str(node.id), pos, cv2.FONT_HERSHEY_SIMPLEX , 1, region_color[i] , 2)
            for neighbor in node.neighbor_nodes.values():
                pos2 = (int(scale/w*neighbor.coord[1]), int(scale/w*neighbor.coord[0]))
                img = cv2.arrowedLine(img, pos, pos2, 0)
        # return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    def get_edges(self):
        if len(self.edges):
            return self.edges.copy()
        else:
            edges=[]
            for i, node in self.nodes.items():
                edges.append([i,i])
                for j in node.neighbor_nodes.keys():
                    edges.append([i,j])
            self.edges = edges
            return edges
        
class Hierarchical_Graph:
    class Node:
        def __init__(self, id_, coord ,level):
            self.id = id_
            self.neighbor_nodes = {}
            self.neighbor_regions_nodes = {}
            self.children = {}
            self.parent = None
            self.level = level
            self.coord = coord
            self.weight = 0
            
        def add_parent(self, parent_node):
            self.parent = parent_node
            parent_node.children[self.id] = self
        
        def add_neighbor(self, neighbor):
            self.neighbor_nodes[neighbor.id] = neighbor
            if neighbor.parent.id != self.parent.id:
                if neighbor.parent.id not in self.neighbor_regions_nodes.keys():
                    self.neighbor_regions_nodes[neighbor.parent.id] = []
                self.neighbor_regions_nodes[neighbor.parent.id].append(neighbor)
                
                if neighbor.parent.id not in self.parent.neighbor_nodes.keys():
                    self.parent.add_neighbor(neighbor.parent)
                    
    def __init__(self, root):
        self.levels={}
        self.levels[0] = {0:root}
        
    def level_to_graph(level):
        pass 
    
    def compute_edges(self, level):
        edges=[]
        for i, node in self.levels[level].nodes.items():
            edges.append([i,i])
            for j in node.neighbor_nodes.keys():
                edges.append([i,j])
        self.levels[level].edges = edges
        
    def get_edges(self, level):
        if len(self.levels[level].edges):
            return self.levels[level].edges
        else:
            self.compute_edges(level)
            return self.levels[level].edges
        
    def grid2graph(self, stencil, level):
        parent_level = self.levels[level-1]
        parent_ids = parent_level.id_map
        parent_nodes = parent_level.nodes
        h, w= parent_ids.shape
        node_ids = np.zeros((h,w))
        node_ids[parent_ids==-1] = -1
        nodes={}
        id_=0
        for i in range(h):
            for j in range(w):
                if  parent_ids[i,j]>=0:
                    node_ids[i,j] = id_
                    node = Hierarchical_Graph.Node(id_, [i,j], level)
                    nodes[id_] = node
                    node.add_parent(parent_nodes[parent_ids[i,j]])
                    id_+=1

        for node in nodes.values():
            y,x = node.coord
            for i,j in stencil:
                if x+i>=0 and x+i<w and y+j>=0 and y+j<h:
                    if node_ids[y+j,x+i]>=0:
                        neighbor = nodes[node_ids[y+j,x+i]]
                        node.add_neighbor(neighbor)
        return nodes, node_ids   