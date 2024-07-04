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
import numpy as np
import ros_numpy
import pickle 

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")

import rospy
def handle(req):
     print("Requested Region ID: "+ str(req.regionID))
     h, cloud = detector.get_region_entropy(req.regionID)
     cloud = cloud.paint_uniform_color([0,0,0])
     msg = get_pc_msg(cloud)
     return PointCloudWithEntropyResponse(msg)
 
def get_pc_msg(cloud):
    points = np.array(cloud.points)
    colors =  np.array(cloud.colors)
    h = np.ones(len(points))*0.465
    pc_array = np.zeros(len(points), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('r', np.uint32),
    ('g', np.uint32),
    ('b', np.uint32),
    ('h', np.float32),
    ])
    pc_array['x'] = points[:,0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
    pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
    pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
    pc_array['h'] = h
    pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
    
    return pc_msg
 
def pointcloud_server():
     rospy.init_node('reference_cloud_server')
     s = rospy.Service('get_reference_cloud_region', PointCloudWithEntropy, handle)
     print("PointCloud server online")
     rospy.spin()
 
if __name__ == "__main__":
    with open(path + '/test/graph.pickle', 'rb') as f:
        graph = pickle.load(f)
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
    for node in graph.pose_nodes.values():
        if not node.local_map == None:
            pc, ref = detector.detect(node, graph.feature_nodes)
    pointcloud_server()