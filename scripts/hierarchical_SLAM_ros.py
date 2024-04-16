#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:27:33 2024

@author: hibad
"""
import rospy
from sensor_msgs.msg import PointCloud2
import tf
import apriltag_EKF_SE3
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy
import numpy as np
from hierarchical_SLAM_SE3 import Graph_SLAM
from Lie import SE3, SO3

np.float = np.float64 
np.set_printoptions(precision=2)

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
    
def get_pose_markers(nodes):
      markers=[]
      for node in nodes.values():
          marker=Marker()
          M=node.M
          p=Pose()
          p.position.x = M[0,3]
          p.position.y = M[1,3]
          p.position.z = M[2,3]
          
          q=tf.transformations.quaternion_from_matrix(M)
          p.orientation.x = q[0]
          p.orientation.y = q[1]
          p.orientation.z = q[2]
          p.orientation.w = q[3]
          

          marker = Marker()
          marker.type = 0
          marker.id = 200 + node.id
          
          marker.header.frame_id = "map"
          marker.header.stamp = rospy.Time.now()
          
          marker.pose.orientation.x=0
          marker.pose.orientation.y=0
          marker.pose.orientation.z=0
          marker.pose.orientation.w=1
          
          
          marker.scale.x = 0.25
          marker.scale.y = 0.05
          marker.scale.z = 0.05
          
          # Set the color
          marker.color.r = 0.0
          marker.color.g = 1.0
          marker.color.b = 1.0
          marker.color.a = 1.0
          marker.pose = p
          markers.append(marker)
          
      return markers
  
def get_landmark_markers(nodes):
    markers=[]
    for node in nodes.values():
        marker=Marker()
        M=node.M
        p=Pose()
        p.position.x = M[0,3]
        p.position.y = M[1,3]
        p.position.z = M[2,3]
        
        q=tf.transformations.quaternion_from_matrix(M)
        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]


    
        marker = Marker()
        marker.type = 0
        marker.id = node.id
        
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        
        marker.pose.orientation.x=0
        marker.pose.orientation.y=0
        marker.pose.orientation.z=0
        marker.pose.orientation.w=1
        
        
        marker.scale.x = 0.5
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Set the color
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.pose = p
        markers.append(marker)
    return markers
    
# def get_factor_markers(graph):
#     P=[]
#     node=graph.nodes
#     for factor in graph.factor:
        
#         mu1=node[edge.node1.id].mu
#         p1=Point()
#         p1.x=mu1[0]
#         p1.y=mu1[1]
#         p1.z=0
#         P.append(p1)
        
        
#         mu2=node[edge.node2.id].mu
#         p2=Point()
#         p2.x=mu2[0]
#         p2.y=mu2[1]
#         if edge.node2.type=="pose":
#             p2.z=0
#         else:
#             p2.z=mu2[2]
#         P.append(p2)


#     marker = Marker()
#     marker.type = 5
#     marker.id = 2

#     marker.header.frame_id = "map"
#     marker.header.stamp = rospy.Time.now()
#     marker.pose.orientation.x=0
#     marker.pose.orientation.y=0
#     marker.pose.orientation.z=0
#     marker.pose.orientation.w=1
#     marker.scale.x = 0.01

    
#     # Set the color
#     marker.color.r = 0.0
#     marker.color.g = 1.0
#     marker.color.b = 1.0
#     marker.color.a = 0.1
    
#     marker.points = P
#     return marker 

def plot_graph(graph, pub):
    markerArray=MarkerArray()
    pose_marker = get_pose_markers(graph.pose_nodes)
    feature_markers = get_landmark_markers(graph.feature_nodes)
  #  factor_marker= get_factor_markers(graph)
    
    markers = feature_markers+pose_marker
 #   feature_markers.append(factor_marker)

    markerArray.markers=markers
    pub.publish(markerArray)
    
    
if __name__ == "__main__":
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    #prior_feature 
    feature_id = 12 
    R_prior=SO3.Exp([0,0,np.pi/2])
    M_prior=np.eye(4)
    M_prior[0:3,0:3]=R_prior
    M_prior[0:3,3]=[-1.714, 0.1067, 0.1188]
    z=SE3.Log(M_prior)
    
    ekf=apriltag_EKF_SE3.EKF(0)
    
    while not feature_id in ekf.landmarks.keys():
        pass
    
    M_feature = ekf.mu[ekf.landmarks[feature_id]]
    
    M_init = M_prior@np.linalg.inv(M_feature)
    print(M_init)
    graph_slam=Graph_SLAM(M_init, ekf)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    
    graph_slam.front_end.add_node(M_prior,"feature", 12)
    graph_slam.front_end.add_prior_factor([], [feature_id],z, np.eye(6)*0.001 , {} ,{feature_id: 0})
    
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
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
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
        rate.sleep()
