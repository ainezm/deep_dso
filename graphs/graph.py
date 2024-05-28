import networkx as nx
import random
import math 
import numpy as np
from tqdm import tqdm 
import scipy.io

def readNetworkRepositoryGraph(absFilePath):

    g = nx.Graph()
    
    edgeList = open(absFilePath, "r")
    for edge in tqdm(edgeList.readlines(), desc = "Recreating Network Repository Graph Edges"):
        if "%" in edge:
            continue 
        
        edge_i = edge.strip() 
        if ',' in edge_i:
            edge_split = edge_i.split(',')
        else:
            edge_split = edge_i.split()
        assert(len(edge_split)==2)
        
        node_a, node_b = edge_split
        node_a, node_b = int(node_a), int(node_b)

        g.add_edge(node_a, node_b)

    #should be an undirected, unweighted, connected graph
    assert(nx.is_directed(g)==False)
    assert(nx.is_weighted(g)==False)
    # print(len(sorted(nx.connected_components(g), key = len, reverse=True)[0]))
    # assert(nx.is_connected(g))
    g = nx.convert_node_labels_to_integers(g, label_attribute=None).copy()
    assert(g.has_node(0))
    
    edgeList.close()
    return g

def generateGeometricGraph(N, average_degree, visualize):
    '''
    N: Number of nodes
    average_degree = Degree of nodes
    '''

    # Probability of sampling an edge between two nodes 
    P = average_degree / N 
    
    # Max distance between two nodes on unit cube to be joined by an edge
    radius = math.sqrt(average_degree / (N*math.pi))

    # Node Positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(N)}

    # Generate connected geometric graph (https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html)
    while True:
        g = nx.random_geometric_graph(N, radius, pos=pos)
        if nx.is_connected(g):
            break
        print("Not Connected, Try Another Random Graph")

    # Assign Edge Weights
    smallest_edge = 1e16
    for u, v in tqdm(list(g.edges), desc = "Assigning Edge Weights"):
        u_coordinate = pos[u]
        v_coordinate = pos[v]

        g[u][v]['weight'] = np.linalg.norm(np.array(u_coordinate) - np.array(v_coordinate)) 
        smallest_edge = min(smallest_edge, g[u][v]['weight'])

    for u, v in tqdm(list(g.edges)):
        g[u][v]['weight'] += random.uniform(0, smallest_edge)/N

    # Print Some Statistics
    print("Random Geometric Graph Generated")
    print(f'Nodes: {g.number_of_nodes()}')
    print(f'Edges: {g.number_of_edges()}')
    print(f'Average Degree: {2.0 * g.number_of_edges() / g.number_of_nodes()}')
    
    if visualize:
        nx.draw(g, pos, node_size=100)
    
    return g