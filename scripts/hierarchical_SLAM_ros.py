#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:27:33 2024

@author: hibad
"""
import threading 
import rospy
from sensor_msgs.msg import PointCloud2
import tf
from apriltag_EKF_ros import EKF_Wrapper
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy
import numpy as np
from graph_SLAM_SE3 import Graph_SLAM
from Lie import SE3
import yaml
import rospkg
from scipy.spatial.transform import Rotation as Rot
from ergodic_inspection.srv import PlaceNode, PlaceNodeResponse, OptimizePoseGraph, OptimizePoseGraphResponse

np.float = np.float64 
np.set_printoptions(precision=2)
is_sim = rospy.get_param("isSim")
rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")
    
class Graph_SLAM_wrapper:
    def __init__(self, tf_br, params, localize_mode  = False):
        self.params = params 
        self.place_node_req = False
        self.lock=threading.Lock()
        self.tf_br = tf_br
        self.factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
        self.pc_pub = rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
        self.thres = params["Graph_SLAM"]["node_threshold"]
        #prior_feature 
        prior = read_prior()
        self.ekf_wrapper = EKF_Wrapper(0, tf_br, params)
        ekf = self.ekf_wrapper.ekf
        intersection = [id_  for id_ in ekf.features.keys() if id_ in prior["children"]]
        while not len(intersection) and not rospy.is_shutdown():
            intersection = [id_  for id_ in ekf.features.keys() if id_ in prior["children"]]
        
        feature_id = intersection[0]
        M_feature = ekf.mu[ekf.features[feature_id]]
        M_prior = prior["children"][feature_id]
        M_init = M_prior@np.linalg.inv(M_feature)
        graph_slam=Graph_SLAM(M_init, localize_mode,params["Graph_SLAM"]["horizon"], 
                              params["Graph_SLAM"]["forgetting_factor"], 
                              params["Graph_SLAM"]["max_iteration"],
                              params["Graph_SLAM"]["step_size"])
        for id_, M_prior in prior["children"].items():
            graph_slam.factor_graph.add_node(M_prior,"feature", id_)
            
        self.reset_ekf(self, M_init, prior["children"].copy())
        if not localize_mode:
            graph_slam.factor_graph.add_prior_factor(prior['z'], prior["cov"] , {} , {"features": prior["idx_map"]})
        self.graph_slam=graph_slam
        rospy.Service('place_node', PlaceNode, self.place_node_server)
        rospy.Service('optimize_pose_graph', OptimizePoseGraph, self.optimize_server)

    def optimize_server(self, req):
        optimized = self.graph_slam.optimize()
        return OptimizePoseGraphResponse(optimized)
        
    def place_node(self, posterior, key_node):
        with self.lock:
            cloud = self.ekf_wrapper.ekf.cloud.copy()
            self.reset_ekf(pose = self.graph_slam.get_node_est()@posterior['mu'][0], get_cloud = key_node)
            self.graph_slam.place_node(posterior, cloud, key_node)
            global_map = self.graph_slam.global_map_assemble(key_only = True)
        pc_msg = pc_to_msg(global_map)
        self.pc_pub.publish(pc_msg)
        
    def reset_ekf(self, robot_pose, features ,get_cloud = False):
        features = self.graph_slam.get_features_est()
        T = np.linalg.inv(robot_pose)
        for id_, M in features.items():
            features[id_] = T@M
        if self.localization_mode:    
            self.ekf_wrapper.reset(self.graph_slam.current_node_id, features, fixed_landmarks = list(features.keys()) ,get_point_cloud=get_cloud)
        else:
            self.ekf_wrapper.reset(self.graph_slam.current_node_id, features, get_point_cloud=get_cloud)

    def place_node_server(self, req):
        print("place keynode")
        # posterior = self.ekf_wrapper.ekf.get_posterior()         
        # node_id = self.place_node(posterior, True)
        self.place_node_req = True
        return  PlaceNodeResponse(True)
    
    # def get_global_map(self):
    #     self.graph_slam.global_map_assemble(key_only = True)
        
    def update(self):
        posterior = self.ekf_wrapper.ekf.get_posterior()         
        _ = self.graph_slam.update(posterior)
        delta = np.linalg.norm(SE3.Log(posterior["mu"][0]))
        if delta >= self.thres or self.place_node_req:
            self.place_node(posterior, self.place_node_req)
            self.place_node_req = False
            
        plot_graph(self.graph_slam.factor_graph, self.factor_graph_marker_pub)
        
        M = self.graph_slam.get_node_est()
        self.tf_br.sendTransform([M[0,3], M[1,3], M[2,3]],
                        tf.transformations.quaternion_from_matrix(M),
                        rospy.Time.now(),
                        "ekf",
                        "map") 
        

            
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
          marker.color.g = 0.0
          marker.color.b = 1.0
          marker.color.a = 1.0
          marker.pose = p
          markers.append(marker)
          
      return markers
  
def get_feature_markers(nodes):
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
        
        
        marker.scale.x = 0.25
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
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
    pose_marker = get_pose_markers(graph.key_pose_nodes)
    feature_markers = get_feature_markers(graph.feature_nodes)
  #  factor_marker= get_factor_markers(graph)
    
    markers = feature_markers+pose_marker
 #   feature_markers.append(factor_marker)

    markerArray.markers=markers
    pub.publish(markerArray)
    
    

def read_prior():
    prior={}
    if is_sim:
        param_path = path + "/resources/sim/"
    else:
        param_path = path +"/resources/real/"
    file = param_path + "/prior_features.yaml"
    with open(file) as stream:
        try:
            features = yaml.unsafe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    z = np.zeros(6*len(features))
    idx_map = {}
    children = {}
    for i, (feature_id, pose) in enumerate(features.items()):
        idx_map[feature_id] = i
        M = np.eye(4)
        M[0:3,0:3] = Rot.from_euler('xyz', pose["orientation"]).as_matrix()
        M[0:3,3] =  pose["position"]
        z[6*i:6*i+6]=SE3.Log(M)
        children[feature_id] = M
        
    prior["z"] = z
    prior["idx_map"] = idx_map
    prior["children"] = children
    prior["cov"] = np.eye(len(z))* 0.0001
    return prior

if __name__ == "__main__":
    import pickle
    if is_sim:
        param_path = path + "/param/sim/"
    else:
        param_path = path +"/param/real/"
        
    with open(param_path+'estimation_param.yaml', 'r') as file:
        params = yaml.safe_load(file)    
    localization_mode = True
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    graph_slam_wrapper = Graph_SLAM_wrapper(br, params,localization_mode)
    

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        graph_slam_wrapper.update()
        # with open('graph.pickle', 'wb') as handle:
        #     pickle.dump(graph_slam_wrapper.graph_slam.factor_graph, handle)
        with open('key_nodes.pickle', 'wb') as handle:
            pickle.dump(graph_slam_wrapper.graph_slam.factor_graph.key_pose_nodes, handle)
        rate.sleep()
