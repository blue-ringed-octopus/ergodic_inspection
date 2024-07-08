#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:52:00 2024

@author: hibad
"""
from map_manager import Map_Manager
import open3d as o3d
from ergodic_inspection.srv import PointCloudWithEntropy, PointCloudWithEntropyResponse
from ergodic_inspection.srv import SetBelief, SetBeliefResponse
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray 
import rospkg
import yaml
import numpy as np
import ros_numpy
import pickle 
import rospy

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")

class Server:
    def __init__(self, map_manager):
        self.map_manager = map_manager
        rospy.init_node('reference_cloud_server')
        rospy.Service('get_reference_cloud_region', PointCloudWithEntropy, self.send_pc)
        rospy.Service('set_entropy', SetBelief, self.set_entropy)

        print("PointCloud server online")
        
    def set_entropy(self, req):
        print("receiving anomaly belief")
        p = req.p.data
        self.map_manager.set_entropy(p)
        print(self.map_manager.h)
        return SetBeliefResponse(True)
    
    def send_pc(self, req):
        if req.regionID == -1:
            print("Requested full workspace")
        else:
            print("Requested Region ID: "+ str(req.regionID))
        h, cloud = self.map_manager.get_region_entropy(req.regionID)
        msg = self.get_pc_msg(cloud,h)
        return PointCloudWithEntropyResponse(msg)
    
    def get_pc_msg(self, cloud, h):
        points = np.array(cloud.points)
        colors =  np.array(cloud.colors)
        normals = np.array(cloud.normals)
        pc_array = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('i', np.float32),
        ('j', np.float32),
        ('k', np.float32),
        ('r', np.uint32),
        ('g', np.uint32),
        ('b', np.uint32),
        ('h', np.float32),
        ])
        pc_array['x'] = points[:,0]
        pc_array['y'] = points[:, 1]
        pc_array['z'] = points[:, 2]
        pc_array['i'] = normals[:,0]
        pc_array['j'] = normals[:, 1]
        pc_array['k'] = normals[:, 2]
        pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
        pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
        pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
        pc_array['h'] = h
        pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
        return pc_msg
      
 
if __name__ == "__main__":
    map_manager = Map_Manager(path)
    Server(map_manager)
    rospy.spin()

