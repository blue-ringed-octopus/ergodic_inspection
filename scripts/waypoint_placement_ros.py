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

import ros_numpy
import numpy as np
from ergodic_inspection.srv import PointCloudWithEntropy, PlanRegion, GetRegion, PlaceNode, OptimizePoseGraph
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
with open(path+'/param/estimation_param.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
class Waypoint_Placement_Wrapper:
    def __init__(self):
        strategy = "ergodic"
        rospy.init_node('waypoint_planner',anonymous=False)
        rospy.wait_for_service('get_reference_cloud_region')
        rospy.wait_for_service('static_map')
        rospy.wait_for_service('get_region')
        rospy.wait_for_service('plan_region')
        rospy.wait_for_service('optimize_pose_graph')
        
        self.get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        self.plan_region = rospy.ServiceProxy('plan_region', PlanRegion)
        self.get_region = rospy.ServiceProxy('get_region', GetRegion)
        self.get_cost_map = rospy.ServiceProxy('static_map', GetMap)
        self.place_node = rospy.ServiceProxy('place_node', PlaceNode)
        self.optimize = rospy.ServiceProxy('optimize_pose_graph', OptimizePoseGraph)

        costmap_msg = self.get_cost_map()
        costmap = process_costmap_msg(costmap_msg)
        self.listener=tf.TransformListener()
        self.listener.waitForTransform(params["EKF"]["optical_frame"],params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransform(params["EKF"]["optical_frame"], params["EKF"]["robot_frame"], rospy.Time(0))
        self.pose = np.array([trans[0], trans[1], np.arctan2(2*(rot[2]), 2*rot[3])])
        T_camera = self.listener.fromTranslationRotation(trans, rot)
        camera_info = rospy.wait_for_message(params["EKF"]["camera_info"], CameraInfo)
        K = np.reshape(camera_info.K, (3,3))
        # K = np.array([[872.2853801540007, 0.0, 604.5],
        #              [0.0, 872.2853801540007, 360.5],
        #              [ 0.0, 0.0, 1.0]])
        w, h = camera_info.width , camera_info.height
        self.next_region = "0"
    
        self.planner = Waypoint_Planner(strategy, costmap, T_camera, K, (w,h))
        self.count = 0 
        
    def get_current_region(self):
        pose_msg = Pose()
        try:
            self.listener.waitForTransform("map",params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform("map", params["EKF"]["robot_frame"], rospy.Time(0))
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
    
    def update(self):
        # try:
            pose, region = self.get_current_region()
            # rospy.wait_for_service('plan_region')
            self.next_region = self.plan_region(region).next_region
            msg = self.get_reference(self.next_region)
            h, region_cloud = decode_msg(msg.ref)
            waypoint = self.planner.get_optimal_waypoint(50, region_cloud, h)
            
            alpha = np.arctan2(waypoint[1]-pose[1], waypoint[0]-pose[0]) - pose[2]
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            if abs(alpha)>np.pi/2:
                intermediate_waypoint = pose.copy()
                intermediate_waypoint += [0.01*np.cos(pose[2]), 0.01*np.sin(pose[2]), alpha]
                # im = self.planner.plot_waypoints([waypoint, intermediate_waypoint])
                # plt.imshow(im, origin="lower")
                # plt.pause(0.05)
                # plt.show()
                navigate2point(intermediate_waypoint)

            self.waypoint = waypoint
            navigate2point(waypoint)
        # except Exception as e: 
        #     print(e)
            self.place_node()
            self.count += 1
        # self.optimize()
    
def decode_msg(msg):
    pc=ros_numpy.numpify(msg)
    x=pc['x'].reshape(-1)
    points=np.zeros((len(x),3))
    points[:,0]=x
    points[:,1]=pc['y'].reshape(-1)
    points[:,2]=pc['z'].reshape(-1)
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
    print("sending rviz arrow")
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
    wrapper = Waypoint_Placement_Wrapper()
    waypoint_thread = threading.Thread(target = plot_waypoint,daemon=True, args = (wrapper,))
    wrapper.update()
    waypoint_thread.start()
    while not rospy.is_shutdown() and wrapper.count<=50:
        wrapper.update()
        
    print("inspection done")