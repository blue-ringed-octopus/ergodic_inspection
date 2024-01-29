#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:26:02 2024

@author: hibad
"""

import numpy as np
np.float = np.float64 
np.set_printoptions(precision=2)
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
import time 
from numpy import sin, cos
from hierarchical_SLAM import *
import tf
import apriltag_EKF


rospack=rospkg.RosPack()


class Anomaly_Detector:
    def __init__(self, mesh):
        self.mesh = mesh
        pc = mesh.sample_points_uniformly(number_of_points=20000, use_triangle_normal=True)
        self.reference = pc 
        self.p_anomaly = np.ones(len(pc.points))
        self.ref_tree=KDTree(np.asarray(pc.points))

    def get_cloud_covariance(self, depth_img):
        n, m = depth_img.shape
        T=self.T_c_to_r[0:3,0:3].copy()@inv(self.K.copy())
        J=[T@np.array([[depth_img[i,j],0,i],
                    [0,depth_img[i,j],j],
                    [0,0,1]]) for i in range(n) for j in range(m)]
    
        cov=np.asarray([j@self.Q[0:3,0:3]@j for j in J])
        return cov
    
    def get_mdist(self, cloud):
        mds=[]
        correspondence=[]
        H=np.asarray(cloud.covariance)
            
        return np.array(mds), correspondence
    
    def detect(self, node):
        t=time.time()

        cloud=deepcopy(node.local_map).transform(node.T)
        point_cov=self.get_cloud_covariance(node.depth)
        sigma_node=node.Cov
        points=np.asarray(cloud.points)
        mu=deepcopy(self.reference)
        theta_node=node.mu[3]
        c=cos(theta_node)
        s=sin(theta_node)
        J_p=np.array([[c, -s,0],
                      [s, c,0],
                      [0,0, 1]])
        
        self.ref_tree.query(points, 1)
        for i in range(len(points)):
            p=points[i]
            J_node = np.array([[1, 0 , -p[1]*c - p[0]*s ],
                               [0, 1, p[0]*c - p[1]*s],
                               [0,0,1]])
            
            sigma=J_node@node.cov@J_node.T+J_p@point_cov[i,:,:]@J_p.T

        print(time.time()-t)

            
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


    
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    ekf=apriltag_EKF.EKF(0)
    graph_slam=Graph_SLAM(np.zeros(3), ekf)
    detector=Anomaly_Detector(mesh)
    marker = get_mesh_marker(mesh_resource)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
    cad_pub = rospy.Publisher("/ref", Marker, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cad_pub.publish(marker)

        optimized=graph_slam.update()
    
     
        plot_graph(graph_slam.front_end)
        
        mu=graph_slam.mu.copy()        
        br.sendTransform([mu[0], mu[1], 0],
                        tf.transformations.quaternion_from_euler(0, 0, mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
    
        if optimized:
            detector.detect(graph_slam.front_end.pose_nodes[0])
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
        rate.sleep()

