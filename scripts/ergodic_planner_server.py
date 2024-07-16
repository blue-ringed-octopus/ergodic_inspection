#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:49:39 2024

@author: hibad
"""

from ergodic_planner import Ergodic_Planner
import rospy
from ergodic_inspection.srv import GetGraphStructure

def parse_graph_msg(msg):
    graph=msg.graph
    nodes = graph.node_ids
    id_map = {}
    for i, node in enumerate(nodes):
        id_map[node] = i
        
    edges=[]
    for edge in graph.edges:
        node1, node2 = edge.split(',')
        edges.append([id_map[node1], id_map[node2]])
    w = graph.weights.data    
    return nodes, edges, w

if __name__ == '__main__':
    rospy.wait_for_service('GetGraphStructure')
    get_graph = rospy.ServiceProxy('GetGraphStructure', GetGraphStructure)
    msg = get_graph(1)
    nodes, edges, w = parse_graph_msg(msg)
    print(nodes)
    print(edges)
    print(w)
    # region_planner = Ergodic_Planner(map_manager.hierarchical_graph.levels[1])
