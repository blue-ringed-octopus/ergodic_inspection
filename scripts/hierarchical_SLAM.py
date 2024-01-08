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
from common_functions import angle_wrapping, v2t, t2v, np2pc
from scipy.linalg import solve_triangular
from scipy.spatial import KDTree
from numpy import sin, cos, arctan2
import open3d as o3d 
from copy import deepcopy
import ros_numpy
np.float = np.float64 

np.set_printoptions(precision=2)

def pose_dist(x1, x2):
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2+10*(x1[2]-x2[2])**2)

class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, mu, node_type):
                self.type=node_type
                self.set_mu(mu)
                self.H=np.eye(3)*0.001
                self.id=node_id
                self.children={}
                self.parents={}
                self.local_map=None
                
            def set_mu(self,mu):
                self.mu=mu.copy()
                self.n=len(self.mu)
                if self.n==3:
                    self.T=v2t([mu[0], mu[1], 0, mu[2]])
                else:
                    self.T=v2t(mu)
                    
        class Edge:
            def __init__(self, node1, node2, Z, omega, edge_type):
                self.node1=node1
                self.node2=node2
                self.Z=Z
                self.omega=omega
                self.type=edge_type
                node1.children[node2.id]={"edge": self, "children": node2}
                node2.parents[node1.id]={"edge": self, "parents": node1}

            # def get_error(self):
            #     e=t2v(np.linalg.inv(self.Z)@(np.linalg.inv(v2t(self.node1.x))@v2t(self.node2.x)))
            #     return e
                
        def __init__(self):
            self.nodes=[]
            self.pose_nodes=[]
            self.edges=[]
            self.feature_nodes={}
        
        def add_node(self, x, node_type, feature_id=None, ):
            i=len(self.nodes)
            node=self.Node(i,x, node_type)
            self.nodes.append(node)
            if node_type=="pose":
                self.pose_nodes.append(node)
            if node_type=="feature":
                self.feature_nodes[feature_id]=node
            return i
        
        def add_edge(self, node_1_id, node_2_id, Z, omega, edge_type="odom"):
            self.edges.append(self.Edge(self.nodes[node_1_id], self.nodes[node_2_id],Z,omega,edge_type))
        
        
    class Back_end:
        def get_pose_jacobian(self, x1, x2, Z):
            ztheta=arctan2(Z[1,0], Z[0,0])
            
            J1=np.array([[-cos(x1[2] + ztheta), -sin(x1[2] + ztheta), x2[1]*cos(x1[2] + ztheta) - x1[1]*cos(x1[2] + ztheta) + x1[0]*sin(x1[2] + ztheta) - x2[0]*sin(x1[2] + ztheta)],
                         [ sin(x1[2] + ztheta), -cos(x1[2] + ztheta), x1[0]*cos(x1[2] + ztheta) - x2[0]*cos(x1[2] + ztheta) + x1[1]*sin(x1[2] + ztheta) - x2[1]*sin(x1[2] + ztheta)],
                         [                    0,                     0,                                                                                                        -1]])

            J2=np.array([[ cos(x1[2] + ztheta), sin(x1[2] + ztheta), 0],
                         [-sin(x1[2] + ztheta), cos(x1[2] + ztheta), 0],
                         [ 0,                    0, 1]     ]        )
            return np.asarray(J1), np.asarray(J2)
        
        def pose_error_function(self, x1,x2,Z):
            return t2v(np.linalg.inv(Z)@(np.linalg.inv(v2t(x1))@v2t(x2)))

        def get_feature_jacobian(self, x1 ,x2):
            
            J1 = np.array([[ cos(x1[2]), sin(x1[2]), x1[1]*cos(x1[2]) - x2[1]*cos(x1[2]) + x2[0]*sin(x1[2]) - x1[0]*sin(x1[2])],
                           [-sin(x1[2]), cos(x1[2]), x2[0]*cos(x1[2]) - x1[0]*cos(x1[2]) + x2[1]*sin(x1[2]) - x1[1]*sin(x1[2])],
                           [          0,          0,                                                                   0],
                           [          0,          0,                                                                   1]])
       
       
            J2 = np.array([[-cos(x1[2]), -sin(x1[2]),  0, 0],
                           [ sin(x1[2]), -cos(x1[2]),  0, 0],
                           [          0,           0, -1, 0],
                           [          0,           0,  0, -1]])
            return J1, J2
        
        def feature_error_function(self, x1,x2,z):
            e = np.array([z[0] - x2[0]*cos(x1[2]) + x1[0]*cos(x1[2]) - x2[1]*sin(x1[2]) + x1[1]*sin(x1[3]),
                          z[1] - x2[1]*cos(x1[2]) + x1[1]*cos(x1[2]) + x2[0]*sin(x1[2]) - x1[0]*sin(x1[3]),
                                                                  z[2] - x2[2],
                          angle_wrapping(z[3]-arctan2(sin(x2[3] - x1[2]), cos(x2[3] - x1[2])))])
            
            return e
        
        def linearize(self,x, edges, idx_map):
            H=np.zeros((len(x), len(x)))
            b=np.zeros(len(x))
            for edge in edges:
                i=idx_map[str(edge.node1.id)]
                j=idx_map[str(edge.node2.id)]
                omega=edge.omega 
                Z=edge.Z
                if edge.type=="odom":
                    A,B=self.get_pose_jacobian(x[i:i+3], x[j:j+3], Z)
                    e=self.pose_error_function(x[i:i+3], x[j:j+3], Z)
                else:
                    A,B=self.get_feature_jacobian(x[i:i+3], x[j:j+4])
                    e=self.feature_error_function(x[i:i+3], x[j:j+4], Z)
                    
                n=A.shape[1] 
                m=B.shape[1] 

                H[i:i+n,i:i+n]+=A.T@omega@A
                H[j:j+m,j:j+m]+=B.T@omega@B
                H[i:i+n,j:j+m]+=A.T@omega@B
                H[j:j+m,i:i+n]+=H[i:i+n,j:j+m].T
                
                b[i:i+n]+=A.T@omega@e
                b[j:j+m]+=B.T@omega@e
            return H,b
        
        def __init__(self):
            pass
        
        def node_to_vector(self, graph):
            idx_map={}
            # x=np.zeros(3*len(graph.nodes))
            x=[]
            for i,node in enumerate(graph.nodes):
               # x[3*i:3*i+3]=node.mu.copy()
               idx_map[str(node.id)]=len(x)
               x=np.concatenate((x, node.mu.copy()))
              # idx_map[str(node.id)]=node.mu.copy().length*i
            return np.array(x), idx_map
        
        def linear_solve(self, A,b):
            L=np.linalg.cholesky(A)
            y=solve_triangular(L,b, lower=True)
            return solve_triangular(L.T, y)
        
        def update_nodes(self, graph,x,H, idx_map):
            for node in graph.nodes:
                idx=idx_map[str(node.id)]
                nodex=x[idx:idx+node.n]
                nodeH=H[idx:idx+node.n,idx:idx+node.n]
                node.set_mu(nodex.copy())
                node.H=nodeH.copy()

            
        def optimize(self, graph):
            x, idx_map= self.node_to_vector(graph)
            H,b=self.linearize(x,graph.edges, idx_map)
            H[0:3,0:3]+=np.eye(3)
            dx=self.linear_solve(H,-b)
            x+=dx
            i=0
            while np.max(dx)>0.001 and i<1000:
                H,b=self.linearize(x,graph.edges, idx_map)
                H[0:3,0:3]+=np.eye(3)
                dx=self.linear_solve(H,-b)
                x+=dx
                i+=1
            self.update_nodes(graph, x,H, idx_map)
            return x, H
            
    def __init__(self, x_init, ekf):
        self.mu=x_init.copy()
        self.ekf=ekf
        self.reset()

    def reset(self):
        self.front_end=self.Front_end()
        self.back_end=self.Back_end()
        self.current_node_id=self.front_end.add_node(self.mu, "pose",[])
        self.omega=np.eye(3)*0.001
        self.global_map={"map":[], "info":[], "tree":None, "anomaly":[]}
        self.feature_tree=None
        # self.costmap=self.anomaly_detector.costmap

        
    def _buid_feature_tree(self):
        loc=[self.front_end.feature_nodes[key].x[0:2].copy() for key in self.front_end.feature_nodes]
        # loc=[node.mu for node in self.front_end.nodes]
        self.feature_tree=KDTree(loc)
        # self.feature_id=[key for key in self.front_end.feature_nodes]
    # def k_nearest_node(self, x,k):        
    #      _, idx = self.node_tree.query(x=x,k=k)
         
         # return deepcopy(self.front_end.nodes[idx])
    # def _loop_closure(self, target, matching_features):
    #     print("loop close")
    #     self.loop_closed=True
    #     current_node=self.front_end.nodes[self.current_node_id]
    #     z1=[]
    #     z2=[]
    #     for feature in matching_features:
    #         z1.append(t2v(current_node.children[feature]["edge"].Z)[0:2])
    #         z2.append(t2v(target.children[feature]["edge"].Z)[0:2])
        
    #     centroid1=np.mean(z1, axis=0)
    #     centroid2=np.mean(z2, axis=0)
    #     q1=z1-centroid1
    #     q2=z2-centroid2

    #     H=q1.T@q2
    #     U, _, V_t = np.linalg.svd(H)
    #     Rot = V_t.T@U.T
    #     dz = centroid2 - Rot@centroid1
    #     Z=np.hstack((Rot, dz.reshape((-1,1))))
    #     Z=np.vstack((Z,[0,0,1]))
    #     omega=np.eye(3)*0.001
    #     self.front_end.add_edge(self.current_node_id,target.id, Z, omega, edge_type="loop_closure")
    #     x, H=self.back_end.optimize(self.front_end)
    #     self.omega=H
        
               
    def _posterior_to_factor(self, mu, sigma,node_to_origin):
        features=self.ekf.landmarks
        for feature_id in features:
              idx=features[feature_id]
              z=mu[idx:idx+4]
              #z=np.append(z,1)
              # print(z)
              # print(node_to_origin)
              # z=node_to_origin@z

              feature_node_id=self.front_end.feature_nodes[feature_id].id
              omega=np.linalg.inv(sigma[idx:idx+4, idx:idx+4])
              omega=(omega+omega.T)/2
              self.front_end.add_edge(self.current_node_id,feature_node_id, z,omega , edge_type="measurement")

    # def search_proximity_nodes(self, pose, radius):
    #     nodes=[deepcopy(node) for node in self.front_end.pose_nodes if np.linalg.norm(t2v(np.linalg.inv(v2t(node.mu))@v2t(pose)))<radius]
    #     return nodes
    
    def search_proximity_features(self, pose, radius):
        # features=[deepcopy(self.front_end.feature_nodes[key]) for key in self.front_end.feature_nodes if np.linalg.norm(pose-self.front_end.feature_nodes[key].x[0:2])<radius]
        if self.feature_tree:
            idx=self.feature_tree.query_ball_point(pose[0:2], radius)
            id_=list(self.front_end.feature_nodes.keys())
            feature=[self.front_end.feature_nodes[id_[i]] for i in idx]
        else:
            feature=[]
        return feature
    
    def _create_new_node(self, sigma, Z):
       # node.local_map=self.ekf.cloud
            
        # points=np.asarray([point["loc"] for point in local_map])
        # info=np.asarray([np.linalg.inv(point["cov"]) for point in local_map])
        
        # if len(points):
        #     cloud=o3d.geometry.PointCloud()
        #     cloud.points=o3d.utility.Vector3dVector(np.concatenate((np.asarray(points), np.zeros(len(points)).reshape(-1,1)), axis=1))
        #     _,_, idx=cloud.voxel_down_sample_and_trace(0.01, 
        #                                                    cloud.get_min_bound(), 
        #                                                    cloud.get_max_bound(), 
        #                                                    False)
            
        #     cov=np.asarray([np.linalg.inv(np.sum(info[i,:,:], axis=0)) for i in idx])
        #     points=np.array([np.sum([info[j,:,:]@points[j] for j in i], axis=0) for i in idx  ])
        #     points=[cov[i]@point for i, point in enumerate(points)]
        points=[]    
        cov=[]
        self.front_end.nodes[self.current_node_id].local_map=self.ekf.cloud
        new_node_id=self.front_end.add_node(self.mu,"pose")
        omega=np.linalg.inv(sigma[0:3, 0:3]+np.eye(3)*0.001)
        self.front_end.add_edge(self.current_node_id,new_node_id, Z, omega)
        self.current_node_id=new_node_id      
        _, H=self.back_end.optimize(self.front_end)
        self.omega=H
        # self._buid_feature_tree()
        
    def occupancy_map(self, pointcloud):
        return 
    
    def _get_map_info(self, node, point):
        theta=node.mu[2]
        x=node.mu[0]
        y=node.mu[1]
        xm=point["loc"][0]
        ym=point["loc"][1]
        # hx=np.array([[cos(theta), -sin(theta), - y*cos(theta) - x*sin(theta)],
        #              [sin(theta),  cos(theta),   x*cos(theta) - y*sin(theta)]])
        hx=np.array([[1, 0, - ym*cos(theta) - xm*sin(theta)],
                     [0,  1,   xm*cos(theta) - ym*sin(theta)]])
        
        H=np.linalg.inv(v2t(node.mu)[0:2,0:2]@point["cov"][0:2, 0:2]@v2t(node.mu)[0:2,0:2].T+hx@np.linalg.inv(node.H)@hx.T)
        return H
    
    def _global_map_assemble(self):
        points=[]
        colors=[]
        for node in self.front_end.pose_nodes[-20:]:
            if not node.local_map == None:
                cloud=deepcopy(node.local_map).transform(node.T)
                points.append(np.array(cloud.points))
                colors.append(np.array(cloud.colors))
        points=np.concatenate(points)  
        colors=np.concatenate(colors)  

        self.global_map = np2pc(points, colors)
        # if len(self.front_end.pose_nodes)%5==0:
        #     o3d.visualization.draw_geometries([self.global_map])
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
                Z=v2t(z)
                x=t2v(node_to_origin@Z)
                self.front_end.add_node(x,"feature", feature_id)
    
    def update(self): 
        optimized=False
        mu=self.ekf.mu.copy()
        sigma=self.ekf.sigma.copy()
        features = self.ekf.landmarks

      #  node_x=self.front_end.nodes[self.current_node_id].mu.copy()
        node_to_origin=self.front_end.nodes[self.current_node_id].T.copy()
        T=v2t([mu[0], mu[1], 0, mu[2]])
        
        pose_global = node_to_origin@T
        mu_r=t2v(pose_global)
        self.mu = [mu_r[0],mu_r[1], mu_r[3] ]
        self.init_new_features(mu, node_to_origin, features)
        delta=t2v(np.linalg.inv(node_to_origin)@pose_global)
        delta[2]*=2
        if np.linalg.norm(delta)>=1.5:
            optimized=True
            self._posterior_to_factor(mu, sigma, node_to_origin)
            self._create_new_node(sigma, T)
            self._global_map_assemble()
            self.ekf.reset(self.current_node_id)
        return optimized

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
      for node in nodes:
          
          mu=node.mu
          p=Point()
          p.x=mu[0]
          p.y=mu[1]
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
        p=Pose()
        p.position.x=x[0]
        p.position.y=x[1]
        p.position.z=x[2]
        
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
    
def get_factor_markers(graph):
    P=[]
    node=graph.nodes
    for edge in graph.edges:
        
        mu1=node[edge.node1.id].mu
        p1=Point()
        p1.x=mu1[0]
        p1.y=mu1[1]
        p1.z=0
        P.append(p1)
        
        
        mu2=node[edge.node2.id].mu
        p2=Point()
        p2.x=mu2[0]
        p2.y=mu2[1]
        if edge.node2.type=="pose":
            p2.z=0
        else:
            p2.z=mu2[2]
        P.append(p2)


    # x=self.landmark["0"]
    marker = Marker()
    marker.type = 5
    marker.id = 2

    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.pose.orientation.x=0
    marker.pose.orientation.y=0
    marker.pose.orientation.z=0
    marker.pose.orientation.w=1
    marker.scale.x = 0.01

    
    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.1
    
    marker.points = P
    return marker 

def plot_graph(graph):
    markerArray=MarkerArray()
    pose_marker = get_pose_markers(graph.pose_nodes)
    feature_markers = get_landmark_markers(graph.feature_nodes)
    factor_marker= get_factor_markers(graph)
    
    feature_markers.append(pose_marker)
    feature_markers.append(factor_marker)

    markerArray.markers=feature_markers
    factor_graph_marker_pub.publish(markerArray)
    
    
if __name__ == "__main__":
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    ekf=apriltag_EKF.EKF(0)
    graph_slam=Graph_SLAM(np.zeros(3), ekf)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)

    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        optimized=graph_slam.update()
    
     
        plot_graph(graph_slam.front_end)
        
        mu=graph_slam.mu.copy()        
        br.sendTransform([mu[0], mu[1], 0],
                        tf.transformations.quaternion_from_euler(0, 0, mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
    
        if optimized:
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
        rate.sleep()
