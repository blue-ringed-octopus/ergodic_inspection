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
import ros_numpy
from sensor_msgs.msg import PointCloud2
from numpy.linalg import inv
from scipy.spatial import KDTree

rospack=rospkg.RosPack()


class Anomaly_Detector:
    def __init__(self, mesh):
        self.mesh = mesh
        pc = mesh.sample_points_uniformly(number_of_points=20000, use_triangle_normal=True)
        self.reference = pc 
        self.p_anomaly = np.ones(len(pc.points))
        self.ref_tree=KDTree(np.asarray(pc.points))

    def get_mdist(self, cloud):
        mds=[]
        correspondence=[]
        H=np.asarray(cloud.covariance)
            
        return np.array(mds), correspondence
    
    def detect(self, node):
        cloud=deepcopy(node.local_map).transform(node.T)
        sigma_node=node.Cov
        points=np.asarray(cloud.points)
        mu=deepcopy(self.reference)

        for i in range(len(points)):
            sigma_p=cloud.covariances
            # omega=inv(sigma_p)
            # D=sqrtm(omega)
            # mu.trasform(D)
            # x=D@points[i]
            
            
def get_mesh_marker(mesh_resource):
    marker=Marker()
    marker.id = 0
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.mesh_resource = mesh_resource
    marker.type = 10
    marker.pose.orientation.x=0
    marker.pose.orientation.y=0
    marker.pose.orientation.z=0
    marker.pose.orientation.w=1
    marker.color.r = 0.2
    marker.color.g = 0.2
    marker.color.b = 0.2
    marker.color.a = 0.5
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    return marker

def pc_to_msg(pc):
    points=np.asarray(pc.points)
    colors=np.asarray(pc.colors)
    pc_array = np.zeros(len(points), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('r', np.uint32),
    ('g', np.uint32),
    ('b', np.uint32),
    ])

    pc_array['x'] = points[:,0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
    pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
    pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
    pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
    
    return pc_msg

if __name__ == "__main__":
    path = rospack.get_path("ergodic_inspection")
    mesh_resource = "file:///" + path + "/resources/ballast.STL"
    
    mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")
    rospy.init_node('cad_pub',anonymous=False)
    cad_pub = rospy.Publisher("/ref", Marker, queue_size = 2)
    rate = rospy.Rate(30) 
    marker = get_mesh_marker(mesh_resource)
    detector=Anomaly_Detector()
    

    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cad_pub.publish(marker)
        rate.sleep()
