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
from ergodic_inspection.srv import PointCloudWithEntropy
import open3d as o3d
from waypoint_placement import Waypoint_Planner
import rospkg 
import pickle
import yaml
import colorsys

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")


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

def talker(coordinates):
    array = PoseArray()
    array.header.frame_id = 'map'
    array.header.stamp = rospy.Time.now()
    pose = Pose()
    pose.position.x = float(coordinates[0])
    pose.position.y = float(coordinates[1])
    pose.orientation.w = float(coordinates[2])
    pose.orientation.z = float(coordinates[3])
    array.poses.append(pose)

    pub = rospy.Publisher('simpleNavPoses', PoseArray, queue_size=100)
    rate = rospy.Rate(1) # 1hz

        #To not have to deal with threading, Im gonna publish just a couple times in the begging, and then continue with telling the robot to go to the points
    count = 0
    while count<1:
        rate.sleep()	
        print("sending rviz arrow")
        pub.publish(array)
        count +=1
def navigate2point(coordinates):
    try:
        simple_move((coordinates[0]),(coordinates[1]),(coordinates[2]),(coordinates[3]))
        talker(coordinates)
        print("goal reached")
    except rospy.ROSInterruptException:
        print ("Keyboard Interrupt")


if __name__ == "__main__":
    from ergodic_inspection.srv import SetBelief

    rospy.wait_for_service('get_reference_cloud_region')
    rospy.wait_for_service('set_entropy')

    with open(path+'/resources/costmap.pickle', 'rb') as handle:
        costmap = pickle.load(handle)  
        
    with open(path+"/resources/region_bounds.yaml") as stream:
        try:
            region_bounds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)    
    T_camera = np.eye(4)
    T_camera[0:3,3]= [0.077, -0.000, 0.218]
    T_camera[0:3,0:3] =  [[0.0019938, -0.1555174, 0.9878311],
                          [-0.999998, -0.000156, 0.0019938],
                          [-0.000156, -0.9878331, -0.1555174]]
    K = np.array([[872.2853801540007, 0.0, 604.5],
                 [0.0, 872.2853801540007, 360.5],
                 [ 0.0, 0.0, 1.0]])
    w, h = 1208, 720
        
    planner = Waypoint_Planner(costmap, region_bounds, T_camera, K, (w,h))
    try:
        region = 1 
        get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        msg = get_reference(-1)
        h, region_cloud = decode_msg(msg.ref)
        set_h = rospy.ServiceProxy('get_reference_cloud_region', SetBelief)
        set_h(np.ones(len(h))*0.1)
        msg = get_reference(region)
        h, region_cloud = decode_msg(msg.ref)
        waypoint = planner.get_optimal_waypoint(region, 50, region_cloud, h)
        print(waypoint)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
