#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:17:49 2023

@author: barc
"""
from numpy import cos,sin, arctan2
import numpy as np
import open3d as o3d
import ros_numpy
np.float = np.float64 

def angle_wrapping(theta):
    return arctan2(sin(theta), cos(theta))


def v2t(x):
    return np.array([[cos(x[3]), -sin(x[3]),0, x[0]],
                     [sin(x[3]), cos(x[3]), 0, x[1]],
                     [0, 0, 1, x[2]],
                      [0, 0, 0,1]])
def t2v(T):
    return np.array([T[0,3], T[1,3], T[2,3], arctan2(T[1,0], T[0,0])])



def np2pc(points, rgb=None):
    '''
    creat point cloud object from list of points
    '''
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(np.asarray(points))
    if not rgb is None:
        cloud.colors=o3d.utility.Vector3dVector(rgb)
    return cloud