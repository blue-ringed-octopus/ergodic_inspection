#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:41:45 2023

@author: barc
"""
import rospy
from sensor_msgs.msg import PointCloud2
import tf
import numpy as np
import apriltag_EKF
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from common_functions import np2pc
from scipy.linalg import solve_triangular
from scipy.spatial import KDTree
from numpy import sin, cos, arctan2
from numpy.linalg import inv, norm
from copy import deepcopy
import ros_numpy
from Lie import SE3, SE2, SO3, SO2

np.float = np.float64 

np.set_printoptions(precision=2)
fr = np.zeros((6,3))
fr[0,0]=1
fr[1,1]=1
fr[5,2]=1
ftag = np.zeros((6,4))
ftag[0,0]=1
ftag[1,1]=1
ftag[2,2] = 1
ftag[5,3] = 1

class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, mu, node_type):
                self.type=node_type
                self.set_mu(mu)
                self.Cov=np.eye(3)*9999999
                self.id=node_id
                self.local_map=None
                self.pruned=False 
                self.depth_img=None
                self.factor=[]
                
            def set_mu(self,mu):
                self.n=len(mu)
                if self.type == "pose":
                    self.M=SE3.Exp([mu[0], mu[1], 0,0,0, mu[2]])
                    self.mu = fr.T@SE3.Log(self.M)
                else:
                    self.M=SE3.Exp([mu[0], mu[1], mu[2],0,0, mu[3]])
                    self.mu = ftag.T@SE3.Log(self.M)

                    
        class Factor:
            def __init__(self, parent_node, child_node, feature_nodes, z, sigma, idx_map):
                self.parent=parent_node
                self.child=child_node
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
                self.pruned=False
                self.n = len(feature_nodes)
                self.idx_map = idx_map
            
            def Jacobian(self):
                pass 
            
        def __init__(self):
            self.nodes=[]
            self.pose_nodes={}
            self.factors=[]
            self.feature_nodes={}
            self.window = 20
            self.current_pose_id = -1
            
        def prune_graph(self):
            pass
        
        def add_node(self, x, node_type, feature_id=None, ):
            i=self.current_pose_id+1
            if node_type=="pose":
                node=self.Node(i,x, node_type)
                self.pose_nodes[i]=node
                self.current_pose_id = i
                if len(self.pose_nodes)>=self.window:
                    self.prun_graph()
                    
            if node_type=="feature":
                node=self.Node(feature_id,x, node_type)
                self.feature_nodes[feature_id]=node
            self.nodes.append(node)                
            return self.current_pose_id
        
        def add_factor(self, parent_id, child_id, feature_ids, Z, sigma, idx_map):
            if  parent_id == None:
                parent = None
            else:
                parent = self.pose_nodes[parent_id]
                
            if  child_id == None:
                child = None
            else:
                child = self.pose_nodes[child_id]
                
            features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
            self.factors.append(self.Factor(parent,child,features ,Z,sigma, idx_map))
                
        
        
    class Back_end:
        def get_pose_jacobian(self, tau_1, tau_2, z_bar):
            J1=-fr.T@SE3.Jl_inv(z_bar)@SE3.Jr(tau_1)@fr
            J2=fr.T@SE3.Jr_inv(z_bar)@SE3.Jr(tau_2)@fr
            return J1, J2
        
      
        def get_feature_jacobian(self, tau_1, tau_2, z_bar):
            J1=-ftag.T@SE3.Jl_inv(z_bar)@SE3.Jr(tau_1)@fr
            J2=ftag.T@SE3.Jr_inv(z_bar)@SE3.Jr(tau_2)@ftag
            return J1, J2
        
        def __init__(self):
            pass
        
        def node_to_vector(self, graph):
            self.pose_idx_map={}
            self.feature_idx_map={}
            x=[]
            for node_id, node in graph.pose_nodes.items():
                if not node.pruned:
                    self.pose_idx_map[node_id]=len(x)
                    x=np.concatenate((x, node.mu.copy()))
            for node_id,node in graph.feature_nodes.items():
                self.feature_idx_map[node_id]=len(x)
                x=np.concatenate((x, node.mu.copy()))
                    
            return np.array(x)
        
        def linearize(self,x, factors):
            H = np.zeros((len(x), len(x)))
            b = np.zeros(len(x))
            for factor in factors:
                idx_map = factor.idx_map.copy()
                omega = factor.omega.copy()
                if not factor.parent == None:
                    F = np.zeros((len(x), 6+factor.n*4))         
                    J = np.zeros((3+factor.n*4,6+factor.n*4))
                    e = np.zeros((3+factor.n*4))
                     
                    idx=self.pose_idx_map[factor.parent.id]
                    F[idx:idx+3,0:3] = np.eye(3)
                    M_r1_inv = inv(factor.parent.M.copy())
                    z = factor.z[0:3].copy()
                    tau_r1 = fr@factor.parent.mu.copy()
                    
                    tau_r2 = fr@factor.child.mu.copy()
                    z_bar = SE3.Log(M_r1_inv@factor.child.M.copy())
                    J1,J2 = self.get_pose_jacobian(tau_r1, tau_r2, z_bar)
                    J[0:3,0:3] = J1
                    J[0:3, 3:6] = J2
                    e[0:3] = z - fr.T@z_bar
                    
                    idx=self.pose_idx_map[factor.child.id]
                    F[idx:idx+3,3:6] = np.eye(3)
    
                    for feature in factor.feature_nodes:
                        i = idx_map[feature.id]
                        tau_r2 = ftag@feature.mu.copy()
                        z = factor.z[i:i+4].copy()
                        z_bar = SE3.Log(M_r1_inv@feature.M.copy())
    
                        J1,J2 = self.get_feature_jacobian(tau_r1, tau_r2, np.array([z[0], z[1], z[2],0,0,z[2]]))
                        J[i:i+4, 0:3] = J1
                        J[i:i+4, 3+i:3+i+4] = J2
                        e[i:i+4] = z - ftag.T@z_bar
                        
                        idx=self.feature_idx_map[feature.id]
                        F[idx:idx+4,i:i+4] = np.eye(4)
                else:
                    J=np.eye(len(factor.z))
                    F = np.zeros((len(x), len(factor.z)))   
                    e = np.zeros(len(factor.z))
                    if not factor.child == None:
                        e[0:3]=factor.child.mu.copy()
                        idx=self.pose_idx_map[factor.child.id]
                        F[idx:idx+3,0:3] = np.eye(3)
                        
                    for feature in factor.feature_nodes:
                        i = idx_map[feature.id]
                        z = factor.z[i:i+4].copy()
                        z_bar = feature.mu.copy()
                        e[i:i+4] = z - z_bar
                        
                        idx=self.feature_idx_map[feature.id]
                        F[idx:idx+4,i:i+4] = np.eye(4)
    

                H+=F@J.T@omega@J@F.T
       
                b+=F@J.T@omega@e
            return H,b
        
        def linear_solve(self, A,b):
            A=(A+A.T)/2
            L=np.linalg.cholesky(A)
            y=solve_triangular(L,-b, lower=True)
            return solve_triangular(L.T, y)
        
        def update_nodes(self, graph,x, cov):
            for node in graph.pose_nodes.values():
                if not node.pruned:
                    idx=self.pose_idx_map[node.id]
                    nodex=x[idx:idx+node.n]
                    nodeCov=cov[idx:idx+node.n,idx:idx+node.n]
                    node.set_mu(nodex.copy())
                    node.H=nodeCov.copy()
                    
            for node in graph.feature_nodes.values():
                idx=self.feature_idx_map[node.id]
                nodex=x[idx:idx+node.n]
                nodeCov=cov[idx:idx+node.n,idx:idx+node.n]
                node.set_mu(nodex.copy())
                node.H=nodeCov.copy()

            
        def optimize(self, graph):
            print("optimizing graph")
            x = self.node_to_vector(graph)
            H,b=self.linearize(x,graph.factors)
            dx=self.linear_solve(H,b)
            x+=dx
            i=0
            self.update_nodes(graph, x,np.zeros(H.shape))
            while np.max(dx)>0.001 and i<1000:
                H,b=self.linearize(x,graph.factors)
                print(i)
                print(H)
                print("det", np.linalg.det(H))
                dx=self.linear_solve(H,b)
                print(dx)
                x+=dx
                i+=1
                self.update_nodes(graph, x,np.zeros(H.shape))

            self.update_nodes(graph, x,inv(H))
            print("optimized")

            return x, H
            
    def __init__(self, x_init, ekf):
        self.optimized = False
        self.mu=x_init.copy()
        self.ekf=ekf
        self.reset()

    def reset(self):
        self.front_end=self.Front_end()
        self.back_end=self.Back_end()
        self.current_node_id=self.front_end.add_node(self.mu, "pose")
        self.omega=np.eye(3)*0.001
        self.global_map={"map":[], "info":[], "tree":None, "anomaly":[]}
        self.feature_tree=None
        
        # self.costmap=self.anomaly_detector.costmap
    
    def _posterior_to_factor(self, mu, sigma):
        self.front_end.nodes[self.current_node_id].local_map=self.ekf.cloud
        self.front_end.nodes[self.current_node_id].cloud_cov=self.ekf.cloud_cov
        new_node_id=self.front_end.add_node(mu[0:3],"pose")

        idx_map=self.ekf.landmarks.copy()
        feature_node_id = idx_map.keys()
       
        self.front_end.add_factor(self.current_node_id,new_node_id,feature_node_id, mu.copy(),sigma.copy(), idx_map)
        self.current_node_id=new_node_id      

        
        
    def occupancy_map(self, pointcloud):
        return 
    
    def _global_map_assemble(self):
        points=[]
        colors=[]
        for node in self.front_end.pose_nodes.values():
            if not node.local_map == None and not node.pruned:
                cloud=deepcopy(node.local_map).transform(node.M)
                points.append(np.array(cloud.points))
                colors.append(np.array(cloud.colors))
        points=np.concatenate(points)  
        colors=np.concatenate(colors)  

        self.global_map = np2pc(points, colors)
     
    def update_costmap(self):
        # image = cv2.flip(cv2.imread("map_actual.png"),0)
        # w=int(image.shape[1]*10)
        # h=int(image.shape[0]*10)
        # image= cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
        # robot_radius=self.robot.radius * 10
        # inflation_radius= 99999999999999999
        # cost_scaling_factor = 0.00001
        # self.costmap.make_cost_map(image,robot_radius, inflation_radius,cost_scaling_factor, self.robot.world.bound)
        # cv2.imshow('test', cv2.flip(self.costmap.cost, 0))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        pass 
    
    def init_new_features(self, mu, node_to_origin, features):
        for feature_id in features:
            if not feature_id in self.front_end.feature_nodes.keys():
                idx=features[feature_id]
                z=mu[idx:idx+4]
                Z=SE3.Exp([z[0], z[1], z[2], 0, 0, z[3]])
                x=SE3.Log(node_to_origin@Z)
                
                self.front_end.add_node(ftag.T@x,"feature", feature_id)
    
    def update(self): 
        if self.optimized:
            self.ekf.reset(self.current_node_id)
            self.optimized=False
            
        mu=self.ekf.mu.copy()
        sigma=self.ekf.sigma.copy()
        features = self.ekf.landmarks.copy()
        node_to_origin=self.front_end.pose_nodes[self.current_node_id].M.copy()
        T=SE3.Exp([mu[0], mu[1], 0, 0, 0, mu[2]])
        
        pose_global = node_to_origin@T
        mu_r=SE3.Log(pose_global)
        self.mu = [mu_r[0],mu_r[1], mu_r[5] ]
        self.init_new_features(mu, node_to_origin, features)
        delta=norm(mu[0:3])
        if delta>=1.5:
            print(sigma)
            self._posterior_to_factor(mu, sigma)
            _, H=self.back_end.optimize(self.front_end)
            self.omega=H
            self._global_map_assemble()
            self.optimized=True

        if np.isnan(self.mu).any():
            rospy.signal_shutdown("nan")


        return self.optimized

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
      P=[]
      for node in nodes.values():
          
          M=node.M
          p=Point()
          p.x=M[0,3]
          p.y=M[1,3]
          p.z=0
          
          P.append(p)

      marker = Marker()
      marker.type = 7
      marker.id = 0

      marker.header.frame_id = "map"
      marker.header.stamp = rospy.Time.now()
      
      marker.pose.orientation.x=0
      marker.pose.orientation.y=0
      marker.pose.orientation.z=0
      marker.pose.orientation.w=1

      
      marker.scale.x = 0.1
      marker.scale.y = 0.1
      marker.scale.z = 0.1
      
      # Set the color
      marker.color.r = 1.0
      marker.color.g = 0.0
      marker.color.b = 0.0
      marker.color.a = 1.0
      
      marker.points = P
      return marker
  
def get_landmark_markers(nodes):
    markers=[]
    for node in nodes.values():
        marker=Marker()
        x=node.mu
        M = SE2.Exp([x[0], x[1], x[3]])
        p=Pose()
        p.position.x = M[0,2]
        p.position.y = M[1,2]
        p.position.z = x[2]
        
        p.orientation.w = cos(x[3]/2)
        p.orientation.x = 0
        p.orientation.y = 0
        p.orientation.z = sin(x[3]/2)

    
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
        marker.color.r = 1.0
        marker.color.g = 0.0
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
    pose_marker = get_pose_markers(graph.pose_nodes)
    feature_markers = get_landmark_markers(graph.feature_nodes)
  #  factor_marker= get_factor_markers(graph)
    
    feature_markers.append(pose_marker)
 #   feature_markers.append(factor_marker)

    markerArray.markers=feature_markers
    pub.publish(markerArray)
    
    
if __name__ == "__main__":
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    ekf=apriltag_EKF.EKF(0)
    graph_slam=Graph_SLAM(np.array([0,0,np.pi]), ekf)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    R=SO3.Exp([0,0,np.pi/2])
    M=np.eye(4)
    M[0:3,0:3]=R
    M[0:3,3]=[-1.714, 0.1067, 0.1188]
    tau=ftag.T@SE3.Log(M)
    graph_slam.front_end.add_node(tau,"feature", 12)
    graph_slam.front_end.add_factor(None, None, [12],tau, np.eye(4)*0.001,{12: 0})
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        optimized=graph_slam.update()
    
     
        plot_graph(graph_slam.front_end, factor_graph_marker_pub)
        
        mu=graph_slam.mu.copy() 
        M = SE2.Exp(mu[0:3])
        br.sendTransform([M[0,2], M[1,2], 0],
                        tf.transformations.quaternion_from_euler(0, 0, mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
    
        if optimized:
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
        rate.sleep()
