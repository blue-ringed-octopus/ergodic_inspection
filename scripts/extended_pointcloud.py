# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:53:37 2024

@author: hibad
"""

class Extended_Pointcloud:
    def __init__(self, pc, cov, depth, rgb, features):
        self.pc = pc
        self.cov = cov
        self.depth = depth
        self.rgb = rgb
        self.features = features