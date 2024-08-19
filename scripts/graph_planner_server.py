#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:49:39 2024

@author: hibad
"""

from graph_planner import Graph_Planner
import rospy
from ergodic_inspection.srv import GetGraphStructure
from ergodic_inspection.srv import PlanRegion, PlanRegionResponse
import numpy as np 
import rospkg
import yaml 

rospack=rospkg.RosPack()
path = rospack.get_path("ergodic_inspection")
with open(path+'/param/control_param.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
class Graph_Planner_Server:
    def __init__(self):
        strategy = params["graph_planner"]['strategy']
        rospy.wait_for_service('GetGraphStructure')
        self.get_graph = rospy.ServiceProxy('GetGraphStructure', GetGraphStructure)
        graph_level = 1
        msg = self.get_graph(graph_level)
        self.id_map, nodes, edges, w = self.parse_graph_msg(msg)
        self.planner = Graph_Planner(nodes, edges, strategy = strategy)
        rospy.Service('plan_region', PlanRegion, self.plan_region_handler)


    def plan_region_handler(self, req):
        msg = self.get_graph(1)
        _, _, _, w = self.parse_graph_msg(msg)
        region = self.id_map[req.current_region]
        next_region, _ = self.planner.get_next_region(w, region)
        return PlanRegionResponse(str(next_region))
    
    def parse_graph_msg(self, msg):
        graph = msg.graph
        nodes = graph.node_ids
        id_map = {}
        for i, node in enumerate(nodes):
            id_map[node] = i
            
        edges=[]
        for edge in graph.edges:
            node1, node2 = edge.split(',')
            edges.append([id_map[node1], id_map[node2]])
        w = np.asarray(graph.weights.data)  
        return id_map, nodes, edges, w

if __name__ == '__main__':
    rospy.init_node('graph_planner',anonymous=False)
    Graph_Planner_Server()
    rospy.spin()