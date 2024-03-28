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
import apriltag_EKF_SE3
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from common_functions import np2pc
from scipy.linalg import solve_triangular
from scipy.spatial import KDTree
from numpy import sin, cos, arctan2
from numpy.linalg import inv, norm, lstsq
from copy import deepcopy
import ros_numpy
from Lie import SE3, SO3

np.float = np.float64 

np.set_printoptions(precision=2)

class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, M, node_type):
                self.type=node_type
                self.M=M
                self.H=np.zeros((6,6))
                self.id=node_id
                self.local_map=None
                self.pruned=False 
                self.depth_img=None
                self.factor={}
               # self.prior_factor = None
                    
        class Factor:
            def __init__(self, id_, parent_node, children_nodes, feature_nodes, z, sigma, idx_map ):
                self.parent=parent_node
                self.children=children_nodes
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
                self.omega=(self.omega.T+self.omega)/2
                self.pruned=False
                self.n = len(feature_nodes)
                self.idx_map = idx_map
                self.id = id_
                
            def get_jacobian(self):
                pass 
            
        def __init__(self):
            self.prior_factor = None
            self.pose_nodes={}
            self.factors={}
            self.feature_nodes={}
            self.window = 2
            self.current_pose_id = -1
            self.current_factor_id = 0
        
        def marginalize(self, node):
            prior = self.prior_factor
            pose_idx_map=prior.idx_map["pose"]
            feature_idx_map=prior.idx_map["features"]
            n = 6 * (len(pose_idx_map) + len(feature_idx_map))
            
            for factor in node.factor.values():
                for id_  in factor.idx_map["pose"].keys():
                    if not id_ in pose_idx_map.keys():
                        pose_idx_map[id_] = n*6
                        n += 1
                           
            for factor in node.factor.values():
               for id_ in factor.idx_map["features"].keys():
                   if not id_ in feature_idx_map.keys():
                       feature_idx_map[id_] = n*6
                       n += 1
                    
            H=np.zeros((6*n,6*n)) 
            b = np.zeros(6*n)
            
            H[0:len(prior.omega),0:len(prior.omega)] = prior.omega
            b[0:len(prior.omega)] = prior.omega@prior.z
            
            
            for id_, factor in node.factor.items():
                J = np.zeros((12+factor.n, 12+factor.n))
                idx_map = factor.idx_map["features"].copy()
                F = np.zeros((6*n, 12+factor.n*6))          #map from factor vector to graph vector
                J = np.zeros((6+factor.n*6,12+factor.n*6)) #map from factor vector to observation
                 
                idx=pose_idx_map[factor.parent.id]
                F[idx:idx+6,0:6] = np.eye(6)
                M_r1 = factor.parent.M.copy()
                M_r1_inv = inv(M_r1)
                M_r2 = factor.children[0].M.copy()
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)@SE3.Jr(SE3.Log(M_r1))
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)@SE3.Jr(SE3.Log(M_r2))
                
                idx=pose_idx_map[factor.children[0].id]
                F[idx:idx+6,6:12] = np.eye(6)

                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    M_f = feature.M.copy()
                    z_bar = SE3.Log(M_r1_inv@M_f)

                    J[i:i+6, 0:6] = -SE3.Jl_inv(z_bar)@SE3.Jr(SE3.Log(M_r1))
                    J[i:i+6, 6+i:6+i+6] = SE3.Jr_inv(z_bar)@SE3.Jr(SE3.Log(M_f))
                    
                    idx = feature_idx_map[feature.id]
                    F[idx:idx+6,6+i:6+i+6] = np.eye(6)
            
              
                H += F@(J.T@factor.omega@J)@F.T
                b += F@J.T@factor.omega@factor.z
        
            cov = inv(H)
            z = cov@b
            
            node_idx = pose_idx_map[node.id]
            idx_range = np.arange(node_idx,node_idx+6)
            cov = np.delete(H,idx_range, 0 )
            cov = np.delete(cov,idx_range, 1 )
            
            pose_idx_map.pop(node.id)
            
            for key, idx in pose_idx_map.items():
                if idx > node_idx:
                    pose_idx_map[key] -=6
                    
            for key, idx in feature_idx_map.items():
                if idx > node_idx:
                    feature_idx_map[key] -=6
                    
            prior.z = np.delete(z,idx_range, 0)
            prior.omega = inv(cov)
                
            
        def prune_graph(self):
            for node in list(self.pose_nodes.values())[:-self.window]:
                self.marginalize(node)
                for id_ in node.factor.keys():
                    self.factors.pop(id_)
                self.pose_nodes.pop(node.id)

        def add_node(self, M, node_type, feature_id=None, ):
            i=self.current_pose_id+1
            if node_type=="pose":
                node=self.Node(i,M, node_type)
                self.pose_nodes[i]=node
                self.current_pose_id = i
                if len(self.pose_nodes)>=self.window:
                    self.prune_graph()
                    
            if node_type=="feature":
                node=self.Node(feature_id,M, node_type)
                self.feature_nodes[feature_id]=node
            return self.current_pose_id
        
        def add_prior_factor(self, children_ids, features_ids, z, sigma, pose_idx_map ,feature_idx_map):
            idx_map={"pose": pose_idx_map, "features": feature_idx_map}
            children = [self.pose_nodes[id_] for id_ in children_ids]
            features = [self.feature_nodes[id_] for id_ in features_ids]
                      
            factor = self.Factor(self.current_factor_id, None,children,features ,z,sigma, idx_map)
            for child in children:
                child.prior_factor = factor
                
            for feature in features:
                feature.prior_factor = factor    
                
            self.prior_factor = factor
                        
        def add_factor(self, parent_id, child_id, feature_ids, z, sigma, feature_idx_map):
            idx_map={"pose": {child_id: 0}, "features": feature_idx_map}
            parent = self.pose_nodes[parent_id]
            child = self.pose_nodes[child_id]
                
            features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
            factor = self.Factor(self.current_factor_id, parent,[child],features ,z,sigma, idx_map)
            self.factors[self.current_factor_id] = factor
            
            parent.factor[self.current_factor_id] = factor
            child.factor[self.current_factor_id] = factor
            
            for feature in features:
                feature.factor[self.current_factor_id] = factor
                
            self.current_factor_id += 1    
        
        
    class Back_end:    
        def __init__(self):
            pass
        
        def node_to_vector(self, graph):
            self.pose_idx_map={}
            self.feature_idx_map={}
            n = 0 
            for node_id, node in graph.pose_nodes.items():
                if not node.pruned:
                    self.pose_idx_map[node_id]=n*6
                    n+=1
                    
            for node_id,node in graph.feature_nodes.items():
                self.feature_idx_map[node_id]=n*6
                n+=1
    
            return n
        
        def linearize(self,n, prior, factors):
            H = np.zeros((6*n, 6*n))
            b = np.zeros(6*n)
            
            J = np.eye(len(prior.z))
            F = np.zeros((6*n, len(prior.z)))   
            e = np.zeros(len(prior.z))
            
            #prior
            idx_map = prior.idx_map["features"].copy()
            omega = prior.omega.copy()
            for child in prior.children:
                z = prior.z[0:6].copy()
                z_bar = SE3.Log(child.M.copy())
                e[0:6] = SE3.Log(SE3.Exp(z - z_bar))
                J[0:6, 0:6] = SE3.Jr_inv(z_bar)
                idx=self.pose_idx_map[child.id]
                F[idx:idx+6,0:6] = np.eye(6)
                
            for feature in prior.feature_nodes:
                i = idx_map[feature.id]
                z = prior.z[i:i+6].copy()
                z_bar = SE3.Log(feature.M.copy())
                J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                idx=self.feature_idx_map[feature.id]
                F[idx:idx+6,i:i+6] = np.eye(6)
                
            H+=F@(J.T@omega@J)@F.T
            b+=F@J.T@omega@e    
            
            for factor in factors.values():
                idx_map = factor.idx_map["features"].copy()
                omega = factor.omega.copy()
                F = np.zeros((6*n, 12+factor.n*6))          #map from factor vector to graph vector
                J = np.zeros((6+factor.n*6,12+factor.n*6)) #map from factor vector to observation
                e = np.zeros(len(factor.z)) #difference between observation and expected observation
                 
                idx=self.pose_idx_map[factor.parent.id]
                F[idx:idx+6,0:6] = np.eye(6)
                z = factor.z[0:6].copy()
                M_r1 = factor.parent.M.copy()
                M_r1_inv = inv(M_r1)
                M_r2 = factor.children[0].M.copy()
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)
                e[0:6] = SE3.Log(SE3.Exp(z - z_bar))
                
                idx=self.pose_idx_map[factor.children[0].id]
                F[idx:idx+6,6:12] = np.eye(6)

                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(M_r1_inv@feature.M.copy())

                    J[i:i+6, 0:6] = -SE3.Jl_inv(z_bar)
                    J[i:i+6, 6+i:6+i+6] = SE3.Jr_inv(z_bar)
                    
                    e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+6,6+i:6+i+6] = np.eye(6)
            
              
                H+=F@(J.T@omega@J)@F.T
                b+=F@J.T@omega@e
    
            return H, b
        
        def linear_solve(self, A,b):
            # A=(A+A.T)/2
            # L=np.linalg.cholesky(A)
            # y=solve_triangular(L,b, lower=True)
            
            # return solve_triangular(L.T, y)
            return lstsq(A,b)[0]
        
        def update_nodes_pose(self, graph,dx):
            for node_id, idx in self.pose_idx_map.items():
                graph.pose_nodes[node_id].M = graph.pose_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
      
            for node_id, idx in self.feature_idx_map.items():
                graph.feature_nodes[node_id].M = graph.feature_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
    
        def update_nodes_cov(self, graph,cov):
            for node_id, idx in self.pose_idx_map.items():
                graph.pose_nodes[node_id].cov = cov[idx:idx+6,idx:idx+6].copy()
            for node_id, idx in self.feature_idx_map.items():    
                graph.feature_nodes[node_id].cov = cov[idx:idx+6,idx:idx+6].copy()

        def optimize(self, graph):
            # with open('graph.pickle', 'wb') as handle:
            #     pickle.dump(graph, handle)
            print("optimizing graph")
            n = self.node_to_vector(graph)
            H,b=self.linearize(n, graph.prior_factor , graph.factors)
            dx=self.linear_solve(H,b)
            i=0
            self.update_nodes_pose(graph, 1*dx.copy())
            while np.max(np.abs(dx))>0.01 and i<10:
                print(i)
                H,b=self.linearize(n, graph.prior_factor , graph.factors)
                dx=self.linear_solve(H,b)
                self.update_nodes_pose(graph, 1*dx.copy())
                i+=1
    
            self.update_nodes_cov(graph, inv(H))
            print("optimized")
            

            return H
            
    def __init__(self, M_init, ekf):
        self.optimized = False
        self.M=M_init.copy()
        self.ekf=ekf
        self.reset()

    def reset(self):
        self.front_end=self.Front_end()
        self.back_end=self.Back_end()
        self.current_node_id=self.front_end.add_node(self.M, "pose")
        self.omega=np.eye(3)*0.001
        self.feature_tree=None
        
        # self.costmap=self.anomaly_detector.costmap
    
    def _posterior_to_factor(self, mu, sigma):
        self.front_end.pose_nodes[self.current_node_id].local_map=self.ekf.cloud.copy()
        new_node_id=self.front_end.add_node(self.M.copy(),"pose")

        
        idx_map=self.ekf.landmarks.copy()
        for key, value in idx_map.items():
            idx_map[key] = value*6
        feature_node_id = idx_map.keys()
        z= np.zeros(6*len(mu))
        J = np.zeros((6*len(mu), 6*len(mu)))
        for i, M in enumerate(mu):
            tau=SE3.Log(M)
            z[6*i:6*i+6]=tau
            J[6*i:6*i+6, 6*i:6*i+6] = SE3.Jr_inv(tau)
        sigma = J@sigma@J.T
        self.front_end.add_factor(self.current_node_id,new_node_id,feature_node_id, z,sigma, idx_map)
        self.current_node_id=new_node_id      

        
        
    def occupancy_map(self, pointcloud):
        return 
    
    def _global_map_assemble(self):
        points=[]
        colors=[]
        for node in self.front_end.pose_nodes.values():
            if not node.local_map == None and not node.pruned:
                if( len(node.local_map["features"])) == 0:
                   dM = node.M.copy() 
                else:
                    dm = np.zeros(6)
                    for feature_id, feature in node.local_map["features"].items():
                        dm  += SE3.Log(self.front_end.feature_nodes[feature_id].M.copy()@inv(feature['M']))
                    dm /= len(node.local_map["features"])
                    dM = SE3.Exp(dm)
                cloud=deepcopy(node.local_map['pc']).transform(dM)
                points.append(np.array(cloud.points))
                colors.append(np.array(cloud.colors))
        points=np.concatenate(points)  
        colors=np.concatenate(colors)  

        self.global_map = np2pc(points, colors)
        self.global_map = self.global_map.voxel_down_sample(0.05)
        
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
    
    def init_new_features(self, mu, Mr, features):
        for feature_id, idx in features.items():
            if not feature_id in self.front_end.feature_nodes.keys():
                Z=mu[idx]
                M=Mr@Z
                self.front_end.add_node(M,"feature", feature_id)
    
    def update(self): 
        if self.optimized:
            self.optimized=False
            
        mu=self.ekf.mu.copy()
        sigma=self.ekf.sigma.copy()
        features = self.ekf.landmarks.copy()
        M=self.front_end.pose_nodes[self.current_node_id].M.copy()
        U=mu[0]
        
        pose_global = M@U
        self.M = pose_global
        self.init_new_features(mu, M, features)
        delta=norm(SE3.Log(mu[0]))
        if delta>=1.5:
            self._posterior_to_factor(mu, sigma)
            self.ekf.reset(self.current_node_id)
            H=self.back_end.optimize(self.front_end)
            self.omega=H
            self._global_map_assemble()
            self.optimized=True

        if np.isnan(self.M).any():
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
        marker.id = 200+ node.id
        
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
    
    ekf=apriltag_EKF_SE3.EKF(0)
    M_init = SE3.Exp([0,0,0,0,0,np.pi])
    graph_slam=Graph_SLAM(M_init, ekf)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    R=SO3.Exp([0,0,np.pi/2])
    M=np.eye(4)
    M[0:3,0:3]=R
    M[0:3,3]=[-1.714, 0.1067, 0.1188]
    z=SE3.Log(M)
    graph_slam.front_end.add_node(M,"feature", 12)
    graph_slam.front_end.add_prior_factor([], [12],z, np.eye(6) , {} ,{12: 0})
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
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
        rate.sleep()
