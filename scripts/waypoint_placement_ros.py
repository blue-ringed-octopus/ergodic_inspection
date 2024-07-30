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
import ros_numpy
import numpy as np
from ergodic_inspection.srv import PointCloudWithEntropy, PlanRegion, GetRegion
from nav_msgs.srv import GetMap
import tf 

import open3d as o3d
from waypoint_placement import Waypoint_Planner
import rospkg 
import pickle
import yaml
import threading  

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")
with open(path+'/param/estimation_param.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
class Waypoint_Placement_Wrapper:
    def __init__(self):
        rospy.init_node('waypoint_planner',anonymous=False)
        rospy.wait_for_service('get_reference_cloud_region')
        rospy.wait_for_service('static_map')
        rospy.wait_for_service('plan_region')
        
        self.get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        self.plan_region = rospy.ServiceProxy('plan_region', PlanRegion)
        self.get_region = rospy.ServiceProxy('get_region', GetRegion)
        self.get_cost_map = rospy.ServiceProxy('static_map', GetMap)
        
        costmap_msg = self.get_cost_map()
        costmap = process_costmap_msg(costmap_msg)
        self.listener=tf.TransformListener()
        self.listener.waitForTransform(params["EKF"]["optical_frame"],params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransform(params["EKF"]["optical_frame"], params["EKF"]["robot_frame"], rospy.Time(0))
        T_camera = self.listener.fromTranslationRotation(trans, rot)

        K = np.array([[872.2853801540007, 0.0, 604.5],
                     [0.0, 872.2853801540007, 360.5],
                     [ 0.0, 0.0, 1.0]])
        w, h = 1208, 720
        self.next_region = "0"
    
        self.planner = Waypoint_Planner(costmap, T_camera, K, (w,h))
        
        
    def get_current_region(self):
        pose = Pose()
        self.listener.waitForTransform("map",params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransform("map", params["EKF"]["robot_frame"], rospy.Time(0))
        pose.position.x = trans[0]
        pose.position.y = trans[1]
        region = self.get_region(pose,1).region
        if region=="-1":
            region = self.next_region
        return region 
    
    def update(self):
        region = self.get_current_region()
        self.next_region = self.plan_region(region).next_region
        msg = self.get_reference(self.next_region)
        h, region_cloud = decode_msg(msg.ref)
        pose = self.planner.get_optimal_waypoint(50, region_cloud, h)
        theta = pose[2]
        self.waypoint = [pose[0], pose[1], np.cos(theta/2), np.sin(theta/2)]
        navigate2point(self.waypoint)
 
    
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

def simple_move(x,y,w,z):
    sac = actionlib.SimpleActionClient('move_base', MoveBaseAction )

    #create goal
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = w
    goal.target_pose.pose.orientation.z = z
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()

    #start listner
    sac.wait_for_server()
    #send goal
    sac.send_goal(goal)
    print("Sending goal:",x,y,w,z)
    #finish
    sac.wait_for_result()
    #print result
    print (sac.get_result())

def talker(waypoint):
    array = PoseArray()
    array.header.frame_id = 'map'
    array.header.stamp = rospy.Time.now()
    pose = Pose()
    pose.position.x = float(waypoint[0])
    pose.position.y = float(waypoint[1])
    pose.orientation.w = float(waypoint[2])
    pose.orientation.z = float(waypoint[3])
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
        simple_move(waypoint[0],waypoint[1],waypoint[2],waypoint[3])
        print("goal reached")
    except rospy.ROSInterruptException:
        print ("Keyboard Interrupt")

def process_costmap_msg(msg):
    map_ = msg.map
    w,h = map_.info.width, map_.info.height
    cost = np.array(map_.data).reshape((h,w))
    resolution = map_.info.resolution
    origin =  [map_.info.origin.position.x, map_.info.origin.position.y]
    return {"costmap": cost, "resolution": resolution,"origin":origin}

def plot_waypoint(wrapper):
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        waypoint = wrapper.waypoint.copy()
        talker(waypoint)
        rate.sleep()	
        
def update(wrapper):
    while not rospy.is_shutdown():
        wrapper.update()

import time
def print_stuff(i):
    for _ in range(10):
        print(i)
        time.sleep(0.01)  
        
if __name__ == "__main__":
    t1 = threading.Thread(target=print_stuff, args = (1,))
    t2 = threading.Thread(target=print_stuff, args = (2,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    # wrapper = Waypoint_Placement_Wrapper()
    # waypoint_thread = threading.Thread(target = plot_waypoint,daemon=True, args = (wrapper,))
    # update_thread = threading.Thread(target = update, args = (wrapper,))
    # wrapper.update()
    # waypoint_thread.start()
    # update_thread.start()
    # rospy.spin()