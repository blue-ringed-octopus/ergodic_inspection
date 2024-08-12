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
from geometry_msgs.msg import Pose

import ros_numpy
from sensor_msgs.msg import PointCloud2
from hierarchical_SLAM_ros import Graph_SLAM_wrapper
from anomaly_detector import Anomaly_Detector
from apriltag_EKF_ros import EKF_Wrapper
import tf
import pickle
import yaml
from ergodic_inspection.srv import PointCloudWithEntropy, SetBelief, GetRegionPointIndex, GetRegion
from std_msgs.msg import Float32MultiArray 
rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")

class Anomaly_Detector_Wrapper:
    def __init__(self, anomaly_thres):
        self.detected_node = []
        self.anomaly_thres = anomaly_thres
        rospy.wait_for_service('get_reference_cloud_region')
        rospy.wait_for_service('set_entropy')
        rospy.wait_for_service('get_region_index')
        rospy.wait_for_service('get_region')

        self.set_h = rospy.ServiceProxy('set_entropy', SetBelief)
        get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        get_region_idx = rospy.ServiceProxy('get_region_index', GetRegionPointIndex)
        self.get_region = rospy.ServiceProxy('get_region', GetRegion)

        msg = get_reference(str(-1))
        reference_cloud = msg_2_pc(msg.ref)
        region_idx = parse_region_idx(get_region_idx())
        # box = reference_cloud.get_axis_aligned_bounding_box()
        # bound = [box.max_bound[0],box.max_bound[1], 0.5 ]
        # box.max_bound = bound

        self.detector = Anomaly_Detector(reference_cloud, region_idx, 0.04)
    
    def detect(self, node, features):
        if node.id not in self.detected_node:
            print("detecting node: ", node.id)

            self.detected_node.append(node.id)
            pose_msg = Pose()
            pose_msg.position.x = node.M[0,3]
            pose_msg.position.y = node.M[1,3]
            rospy.wait_for_service('get_region')
            region = self.get_region(pose_msg,1).region
            
            p, idx = self.detector.detect(node, features, region)
            
            if len(p) != 0:                
                msg = Float32MultiArray()
                msg.data = p
                # try:
                self.set_h(idx.astype(np.uint64), msg)
            # except:
            #     print("failed to send entropy")
                

# def get_mesh_marker(mesh_resource):
#     marker=Marker()
#     marker.id = 0
#     marker.header.frame_id = "map"
#     marker.header.stamp = rospy.Time.now()
#     marker.mesh_resource = mesh_resource
#     marker.type = 10
#     marker.pose.orientation.x=0
#     marker.pose.orientation.y=0
#     marker.pose.orientation.z=0
#     marker.pose.orientation.w=1
#     marker.color.r = 0.2
#     marker.color.g = 0.2
#     marker.color.b = 0.2
#     marker.color.a = 0.5
#     marker.scale.x = 1
#     marker.scale.y = 1
#     marker.scale.z = 1
#     return marker

def pc_2_msg(cloud):
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

def msg_2_pc(msg):
    pc=ros_numpy.numpify(msg)
    x=pc['x'].reshape(-1)
    points=np.zeros((len(x),3))
    points[:,0]=x
    points[:,1]=pc['y'].reshape(-1)
    points[:,2]=pc['z'].reshape(-1)
    
    normals = np.zeros(points.shape)
    normals[:,0]=pc['i']
    normals[:,1]=pc['j']
    normals[:,2]=pc['k']
    
    pc=ros_numpy.point_cloud2.split_rgb_field(pc)
    rgb=np.zeros((len(x),3))
    rgb[:,0]=pc['r'].reshape(-1)
    rgb[:,1]=pc['g'].reshape(-1)
    rgb[:,2]=pc['b'].reshape(-1)

    # p = {"points": points, "colors": np.asarray(rgb/255), "h": h}
    # print(h)
    p=o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    p.colors = o3d.utility.Vector3dVector(np.asarray(rgb/255))
    p.normals = o3d.utility.Vector3dVector(normals)
    return p

def parse_region_idx(msg):
    region_idx={}
    for region in msg.region_idx:
        id_ = region.region_id
        region_idx[id_] = np.array(region.idx)
    return region_idx  
  
if __name__ == "__main__":
    localization_mode = True
    anomaly_thres = 0.04

    detector_wrapper = Anomaly_Detector_Wrapper(anomaly_thres)
    
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    graph_slam_wrapper = Graph_SLAM_wrapper(br, localization_mode)
       
    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        graph_slam_wrapper.update() 
     
        features = graph_slam_wrapper.graph_slam.factor_graph.feature_nodes
        if len(graph_slam_wrapper.graph_slam.factor_graph.key_pose_nodes)> 1: 
            for node in list(graph_slam_wrapper.graph_slam.factor_graph.key_pose_nodes.values())[:-1]:  
               detector_wrapper.detect(node, features)
               del graph_slam_wrapper.graph_slam.factor_graph.key_pose_nodes[node.id]


        rate.sleep()

