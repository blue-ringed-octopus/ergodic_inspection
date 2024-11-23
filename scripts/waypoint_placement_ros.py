#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:04:29 2024

@author: hibad
"""

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import CameraInfo
import numpy as np
np.float = np.float64 
import ros_numpy
from ergodic_inspection.srv import PointCloudWithEntropy, PlanRegion, GetRegion, PlaceNode, OptimizePoseGraph
from ergodic_inspection.srv import GetCandidates
from nav_msgs.srv import GetMap
import tf 

import open3d as o3d
from waypoint_placement import Waypoint_Planner
import rospkg 
import pickle
import yaml
import threading  
import time
import matplotlib.pyplot as plt 

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")
# with open(path+'/param/control_param.yaml', 'r') as file:
#     control_params = yaml.safe_load(file)
# with open(path+'/param/estimation_param.yaml', 'r') as file:
#     est_params = yaml.safe_load(file) 
       
class Waypoint_Placement_Wrapper:
    def __init__(self, ctrl_params, est_params, edge_waypoints=None):
        self.running = True
        self.edge_waypoints = edge_waypoints
        self.ctrl_params = ctrl_params
        self.est_params = est_params
        self.inspection_steps = 15
        self.waypoints_per_region = 2
        
        strategy = ctrl_params["waypoint_placement"]['strategy']
        if strategy == "ergodic":
            self.horizon = ctrl_params["graph_planner"]["horizon"]
        elif strategy == "random":
            self.horizon = np.inf
        elif strategy == "greedy":
            self.horizon = 1
        
        rospy.init_node('waypoint_planner',anonymous=False)
        rospy.wait_for_service('get_reference_cloud_region')
        rospy.wait_for_service('static_map')
        rospy.wait_for_service('get_region')
        rospy.wait_for_service('plan_region')
        rospy.wait_for_service('optimize_pose_graph')
        rospy.wait_for_service('get_anomaly_candidates')

        self.get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        self.plan_region = rospy.ServiceProxy('plan_region', PlanRegion)
        self.get_region = rospy.ServiceProxy('get_region', GetRegion)
        self.get_cost_map = rospy.ServiceProxy('static_map', GetMap)
        self.place_node = rospy.ServiceProxy('place_node', PlaceNode)
        self.optimize = rospy.ServiceProxy('optimize_pose_graph', OptimizePoseGraph)
        self.get_candidates = rospy.ServiceProxy('get_anomaly_candidates', GetCandidates)

        costmap_msg = self.get_cost_map()
        costmap = process_costmap_msg(costmap_msg)
        self.listener=tf.TransformListener()
        self.listener.waitForTransform(est_params["EKF"]["robot_frame"],est_params["EKF"]["optical_frame"],rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransform(est_params["EKF"]["robot_frame"],est_params["EKF"]["optical_frame"] , rospy.Time(0))
        self.pose = np.array([trans[0], trans[1], np.arctan2(2*(rot[2]), 2*rot[3])])
        T_camera = self.listener.fromTranslationRotation(trans, rot)
        camera_info = rospy.wait_for_message(est_params["EKF"]["camera_info"], CameraInfo)
        K = np.reshape(camera_info.K, (3,3))

        w, h = camera_info.width , camera_info.height
        self.next_region = "0"
    
        self.planner = Waypoint_Planner(strategy, costmap, T_camera, K, (w,h))
        self.step = 0 
        
    def get_current_region(self):
        pose_msg = Pose()
        try:
            self.listener.waitForTransform("map",self.est_params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform("map", self.est_params["EKF"]["robot_frame"], rospy.Time(0))
            pose_msg.position.x = trans[0]
            pose_msg.position.y = trans[1]
            rospy.wait_for_service('get_region')
            region = self.get_region(pose_msg,1).region
            pose = np.array([ trans[0], trans[1], np.arctan2(2*(rot[2]), 2*rot[3])])
            if region=="-1":
                region = self.next_region
        except:
            region = self.next_region
            pose = self.waypoint
        return pose, region 
    
    def navigate_intermediate_waypoint(self, p1, p2):
        alpha = np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) - p1[2]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        if abs(alpha)>np.pi/2:
            intermediate_waypoint = p1.copy()
            intermediate_waypoint += [0.01*np.cos(p2[2]), 0.01*np.sin(p2[2]), alpha]
            navigate2point(intermediate_waypoint)
      
    def inspect(self):
        pose, region = self.get_current_region()
        # rospy.wait_for_service('plan_region')
        if self.step==0:
            self.plan_region(region, True)
            self.next_region = region
        elif self.step%self.horizon:
            self.next_region = self.plan_region(region, False).next_region
        else:
            self.next_region = self.plan_region(region, True).next_region
            
        
            
        msg = self.get_reference(self.next_region)
        h, region_cloud = decode_msg(msg.ref)
        waypoints = []
        for _ in range(self.waypoints_per_region):
            waypoint = self.planner.get_optimal_waypoint(1000, region_cloud, h)
            waypoints.append(waypoint)
            
        self.navigate_intermediate_waypoint(pose, waypoints[0])
        
        if self.next_region in self.edge_waypoints[region].keys():
            edge_waypoint = self.edge_waypoints[region][self.next_region]
            edge_waypoint = [edge_waypoint["x"], 
                             edge_waypoint["y"], 
                             edge_waypoint["z"],
                             edge_waypoint["w"],]
            simple_move(edge_waypoint)
            
        for waypoint in waypoints:   
            self.waypoint = waypoint
            navigate2point(waypoint)
            self.place_node()

    # except Exception as e: 
    #     print(e)
        self.step += 1     
        
    def collect_image(self):
        msg = self.get_candidates()
        self.candidates = decode_candidates(msg)
        
        
    def update(self):
        print("step: ", self.step)
        if self.step<=self.inspection_steps:
            print("inspecting")
            self.inspect()
        else:
            print("collecting images")  
            self.collect_image()
            self.running = False
            
def decode_candidates(msg):
    candidates = {}
    for region in msg.region_candidates:
        id_ = region.region_id
        n = region.num_candidates
        if n>0:
            points = np.array(region.points.data)
            points = points.reshape((n,3))
            candidates[id_] = points
        else:
            candidates[id_] = np.array([])
    return candidates     

    
def decode_msg(msg):
    pc=ros_numpy.numpify(msg)
    x=pc['x'].reshape(-1)
    points=np.zeros((len(x),3))
    points[:,0]=x
    points[:,1]=pc['y'].reshape(-1)
    points[:,2]=pc['z'].reshape(-1)
    
    normals = np.zeros((len(x),3))
    normals[:,0]=pc['i'].reshape(-1)
    normals[:,1]=pc['j'].reshape(-1)
    normals[:,2]=pc['k'].reshape(-1)

    pc=ros_numpy.point_cloud2.split_rgb_field(pc)
    rgb=np.zeros((len(x),3))
    rgb[:,0]=pc['r'].reshape(-1)
    rgb[:,1]=pc['g'].reshape(-1)
    rgb[:,2]=pc['b'].reshape(-1)
    h = pc["h"]

    # p = {"points": points, "colors": np.asarray(rgb/255), "h": h}
    # print(h)
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(points)
    p.colors=o3d.utility.Vector3dVector(np.asarray(rgb/255))
    p.normals = o3d.utility.Vector3dVector(normals)
    return h, p

def simple_move(waypoint):
    sac = actionlib.SimpleActionClient('move_base', MoveBaseAction )

    #create goal
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = waypoint[0]
    goal.target_pose.pose.position.y = waypoint[1]
    
    goal.target_pose.pose.orientation.z = waypoint[2]
    goal.target_pose.pose.orientation.w = waypoint[3]
    
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()

    #start listner
    sac.wait_for_server()
    #send goal
    sac.send_goal(goal)
    print("Sending goal:",waypoint)
    #finish
    sac.wait_for_result()
    #print result
    print (sac.get_result())

def talker(waypoint):
    array = PoseArray()
    array.header.frame_id = 'map'
    array.header.stamp = rospy.Time.now()
    pose = Pose()
    theta = waypoint[2]
    pose.position.x = float(waypoint[0])
    pose.position.y = float(waypoint[1])
    
    pose.orientation.z = float(np.sin(theta/2))
    pose.orientation.w = float(np.cos(theta/2))
    array.poses.append(pose)

    pub = rospy.Publisher('simpleNavPoses', PoseArray, queue_size=100)
     # 1hz

        #To not have to deal with threading, Im gonna publish just a couple times in the begging, and then continue with telling the robot to go to the points
    # count = 0
    pub.publish(array)
    # print("sending rviz arrow")
    # while count<10:
    #     rate.sleep()	
    #     print("sending rviz arrow")
    #     pub.publish(array)
    #     count +=1
        
# def talker(waypoint):
#     pose = Pose()
#     # pose.header.frame_id = 'map'
#     # pose.header.stamp = rospy.Time.now()
#     pose.position.x = float(waypoint[0])
#     pose.position.y = float(waypoint[1])
#     pose.orientation.w = float(waypoint[2])
#     pose.orientation.z = float(waypoint[3])

#     pub = rospy.Publisher('/move_base_simple/goal', Pose, queue_size=100)
#     rate = rospy.Rate(1) # 1hz

#         #To not have to deal with threading, Im gonna publish just a couple times in the begging, and then continue with telling the robot to go to the points
#     count = 0
#     pub.publish(pose)
#     while count<10:
#         rate.sleep()	
#         print("sending rviz arrow")
#         pub.publish(pose)
#         count +=1    
    
def navigate2point(waypoint):
    try:
        theta = waypoint[2]
        simple_move([waypoint[0],waypoint[1],np.sin(theta/2),np.cos(theta/2)])
        print("goal reached")
    except rospy.ROSInterruptException:
        print ("Keyboard Interrupt")

def process_costmap_msg(msg):
    map_ = msg.map
    w,h = map_.info.width, map_.info.height
    cost = np.array(map_.data).reshape((h,w))
    resolution = map_.info.resolution
    origin =  [map_.info.origin.position.x, map_.info.origin.position.y]
    return {"costmap": cost.T, "resolution": resolution,"origin":origin}

def plot_waypoint(wrapper):
    while not rospy.is_shutdown():
        waypoint = wrapper.waypoint.copy()
        talker(waypoint)
        time.sleep(1)	
               
if __name__ == "__main__":
    is_sim = rospy.get_param("isSim")
    save_dir= rospy.get_param("save_dir")

    if is_sim:
        param_path = path + "/param/sim/"
        resource_path =  path + "/resources/sim/"
    else:
        param_path = path +"/param/real/"
        resource_path =  path + "/resources/real/"

    with open(param_path+'control_param.yaml', 'r') as file:
        crtl_params = yaml.safe_load(file) 
    with open(param_path+'estimation_param.yaml', 'r') as file:
        est_params = yaml.safe_load(file) 
        
    with open(resource_path+'edge_waypoints.yaml', 'r') as file:
        edge_waypoints = yaml.safe_load(file)  
        
    wrapper = Waypoint_Placement_Wrapper(crtl_params, est_params, edge_waypoints)
    waypoint_thread = threading.Thread(target = plot_waypoint,daemon=True, args = (wrapper,))
    wrapper.update()
    waypoint_thread.start()
    while not rospy.is_shutdown() and wrapper.running:
        wrapper.update()
    candidates = wrapper.candidates.copy()
    
    for region, candidate in candidates.items():
        candidates[region] = candidate.tolist()
        
    with open(save_dir+'anomaly_candidates.yaml', 'w') as file:
        yaml.safe_dump(candidates, file)    
        
    print("inspection done")