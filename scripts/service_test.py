#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:53:01 2024

@author: hibad
"""


# from __future__ import print_function
import open3d as o3d
from anomaly_detector import Anomaly_Detector
from ergodic_inspection.srv import PointCloudWithEntropy, PointCloudWithEntropyResponse
from sensor_msgs.msg import PointCloud2
import rospkg
import yaml
rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")

import rospy
def handle_add_two_ints(req):
     print("Requested Region ID: "+ str(req.regionID))
     msg = PointCloud2()
     return PointCloudWithEntropyResponse(msg)
 
def pointcloud_server():
     rospy.init_node('reference_cloud_server')
     s = rospy.Service('get_reference_cloud_region', PointCloudWithEntropy, handle_add_two_ints)
     print("PointCloud server online")
     rospy.spin()
 
if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")
    with open(path+"/resources/region_bounds.yaml") as stream:
        try:
            region_bounds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    box = mesh.get_axis_aligned_bounding_box()
    bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
    box.max_bound = bound

    global detector
    detector = Anomaly_Detector(mesh, box,region_bounds,0.02)
    pointcloud_server()