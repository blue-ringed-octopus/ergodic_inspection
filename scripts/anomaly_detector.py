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
from scipy.linalg import sqrtm
from visualization_msgs.msg import Marker, MarkerArray

rospack=rospkg.RosPack()

class Anomaly_Detector:
    def __init__(self):
        path = rospack.get_path("ergodic_inspection")
        mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")
        pc = mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=True)
      #  o3d.visualization.draw([pc])
        self.reference = pc 
        
    def get_mdist(self, cloud):
        mds=[]
        correspondence=[]
        H=np.asarray(cloud.covariance)
            
        return np.array(mds), correspondence
    
    def detect(self, node):
        cloud=deepcopy(node.local_map).transform(node.T)
        
    
if __name__ == "__main__":
    rospy.init_node('cad_pub',anonymous=False)
    cad_pub = rospy.Publisher("/ref", Marker, queue_size = 2)
    rate = rospy.Rate(30) 

    path = rospack.get_path("ergodic_inspection")
    detector=Anomaly_Detector()
    marker=Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.mesh_resource = path+"file:///resources/ballast.STL"
    marker.type = 10
    marker.pose.orientation.x=0
    marker.pose.orientation.y=0
    marker.pose.orientation.z=0
    marker.pose.orientation.w=1
    while not rospy.is_shutdown():
        cad_pub.publish(marker)
        rate.sleep()
