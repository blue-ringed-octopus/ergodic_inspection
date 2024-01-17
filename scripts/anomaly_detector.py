#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:26:02 2024

@author: hibad
"""

import numpy as np
import rospy
import open3d as o3d
from copy import deepcopy
# from numba import cuda 
import rospkg

rospack=rospkg.RosPack()

class Anomaly_Detector:
    def __init__(self):
        path = rospack.get_path("ergodic_inspection")
        mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")
        pc = o3d.geometry.sample_points_uniformly(mesh, number_of_points=1000)
        o3d.visualization.draw([pc])
        self.reference = pc 
    
    def detect(self, node):
        cloud=deepcopy(node.local_map).transform(node.T)
        
    
if __name__ == "__main__":
    detector=Anomaly_Detector()
    