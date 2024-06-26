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
import rospkg
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy
from sensor_msgs.msg import PointCloud2
from hierarchical_SLAM_SE3 import Graph_SLAM
from hierarchical_SLAM_ros import plot_graph, pc_to_msg, initialize_graph_slam
from anomaly_detector import Anomaly_Detector
import apriltag_EKF_SE3
import tf
import cv2
import pickle
import yaml

rospack=rospkg.RosPack()

            
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

def get_ref_pc(cloud):
    points = np.array(cloud.points)
    colors =  np.array(cloud.colors)
    
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
    thres = 0.02
    path = rospack.get_path("ergodic_inspection")
    mesh_resource = "file:///" + path + "/resources/ballast.STL"
    
    mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")

   
    
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    ekf=apriltag_EKF_SE3.EKF(0)
    graph_slam = initialize_graph_slam(ekf, localize_mode = True)
    box = mesh.get_axis_aligned_bounding_box()
    bound = [box.max_bound[0],box.max_bound[1], 0.7 ]
    box.max_bound = bound

    detector = Anomaly_Detector(mesh, box,0.02)
    marker = get_mesh_marker(mesh_resource)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
    ref_pc_pub=rospy.Publisher("/pc_ref", PointCloud2, queue_size = 2)
    cad_pub = rospy.Publisher("/ref", Marker, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cad_pub.publish(marker)

        optimized=graph_slam.update()
        
     
        plot_graph(graph_slam.front_end, factor_graph_marker_pub)
        
        M=graph_slam.M.copy() 
        br.sendTransform([M[0,3], M[1,3], M[2,3]],
                        tf.transformations.quaternion_from_matrix(M),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
    

        if optimized:
            graph_slam.global_map_assemble()
            node_id  = list(graph_slam.front_end.pose_nodes.keys())[-2]
            pc, ref = detector.detect(graph_slam.front_end.pose_nodes[node_id], graph_slam.front_end.feature_nodes)
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
            ref_pc = get_ref_pc(ref)
            ref_pc_pub.publish(ref_pc)

        rate.sleep()

