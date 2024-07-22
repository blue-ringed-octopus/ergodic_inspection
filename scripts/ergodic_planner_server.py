#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:49:39 2024

@author: hibad
"""

from ergodic_planner import Ergodic_Planner
import rospy
from ergodic_inspection.srv import GetGraphStructure
from ergodic_inspection.srv import PlanRegion, PlanRegionResponse

import numpy as np 

class Ergodic_Planner_Server:
    def __init__(self):
        rospy.wait_for_service('GetGraphStructure')
        get_graph = rospy.ServiceProxy('GetGraphStructure', GetGraphStructure)
        msg = get_graph(1)
        nodes, edges, w = self.parse_graph_msg(msg)
        self.planner = Ergodic_Planner(nodes, edges)
        rospy.Service('plan_region', PlanRegion, self.plan_region_handler)


    def plan_region_handler(self, req):
        w = req.h.data
        region = req.h.current_region
        next_region = self.planner.get_next_region(w, region)
        return PlanRegionResponse(str(next_region))
    
    def parse_graph_msg(self, msg):
        graph=msg.graph
        nodes = graph.node_ids
        id_map = {}
        for i, node in enumerate(nodes):
            id_map[node] = i
            
        edges=[]
        for edge in graph.edges:
            node1, node2 = edge.split(',')
            edges.append([id_map[node1], id_map[node2]])
        w = np.asarray(graph.weights.data)  
        return nodes, edges, w

if __name__ == '__main__':
    rospy.init_node('ergodic_planner',anonymous=False)
    Ergodic_Planner_Server()
    rospy.spin()