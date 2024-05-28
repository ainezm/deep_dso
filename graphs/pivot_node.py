import networkx as nx
from pyvis import network as net
import random
import math 
import numpy as np
from tqdm import tqdm 

'''
Example Code: 

# Generating Graph
from graphs.graph import readNetworkRepositoryGraph 
g = readNetworkRepositoryGraph(absFilePath = os.path.abspath("./graphs/network_repository/inf-openflights.edges"))

# Generating Pivots
pivots = pivot_node.generatePivots(g, TRAINING_SET_SIZE = 1000, visualize = False)
'''

def generateReplacementPaths(g, TRAINING_SET_SIZE, pos):

    # Simulate Node Failures
    replacement_paths = {}
    replacement_paths_lengths = {}
    successful_samples = 0
    
    pbar = tqdm(total = TRAINING_SET_SIZE, desc = "Simulating Node Failures for Replacement Paths", position = pos)
    while successful_samples < TRAINING_SET_SIZE:
      s = random.choice(list(g.nodes))
      t = random.choice(list(g.nodes))
      f_n = random.choice(list(g.nodes))

      # Connected by a single edge 
      if s==t or s==f_n or t==f_n or nx.shortest_path_length(g, s, t) < 3: #check defintion of pivot 
        continue 

      # Already in Replacement Paths
      if (s, t, f_n) in replacement_paths:
        continue

      # Replacement Path for s -> t if edge e fails
      original_edges = [k for k in g.edges(f_n)]
      g.remove_node(f_n)

      if nx.has_path(g, s, t):
        replacement_paths[(s,t,f_n)] = [p for p in nx.all_shortest_paths(g, s, t)]
        replacement_paths_lengths[(s,t,f_n)] = nx.shortest_path_length(g, s, t)

        successful_samples += 1
        pbar.update(1)
      
      g.add_edges_from(original_edges)

    pbar.close()
    return replacement_paths, replacement_paths_lengths

def visualizeNodeRemoval(g, original_path, failing_node, replacement_path, pivot = None):
  '''
  Inputs:
  - g: Graph object
  - original_path: list of nodes
  - replacement_path: list of nodes

  Process:
  Visualize the subgraph of g which contains the original shortest path from s to t (red),
  the replacement path (green), and the shortest paths (for every node in the replacement 
  path a) from s to a and a to t (black). 
  ''' 

  s = original_path[0]
  t = original_path[-1]
  assert(original_path[0] == replacement_path[0] and original_path[-1] == replacement_path[-1])

  # Get all nodes to visualize
  subgraph_nodes = set()
  subgraph_nodes.update(original_path)
  subgraph_nodes.update(replacement_path)
  
  for a in replacement_path:
    for i in nx.shortest_path(g, s, a):
      subgraph_nodes.add(i)

    for i in nx.shortest_path(g, a, t):
      subgraph_nodes.add(i)
    
  print(f"There are {len(subgraph_nodes)} nodes in our subgraph")

  # Obtain the Subgraph
  k = g.subgraph(list(subgraph_nodes))

  # Visualize
  
  # plt.figure(figsize=(9, 9))
  # # nx.draw(k, pos = nx.circular_layout(k), edgelist = edges, edge_color = colors)
  # nx.draw(k, pos=nx.spectral_layout(k), with_labels=True, node_size=100, node_color="skyblue", alpha=0.5, linewidths=40)
  # # nx.draw_networkx_edge_labels(k, pos=nx.spring_layout(k))

  # plt.axis('off')
  # plt.show()

  nt = net.Network(notebook = True)

  for n in subgraph_nodes:
    if n == pivot:
      color = 'yellow'
    elif n == failing_node:
      color = 'purple'
    elif n in replacement_path:
      color = 'green'
    elif n in original_path:
      color = 'red'
    else:
      color = 'blue'

    nt.add_node(n, color = color, label = str(n))
  
  for u, v in k.edges():
    if u in original_path and v in original_path and abs(original_path.index(u) - original_path.index(v))==1:
      color = 'red'
    elif u in replacement_path and v in replacement_path and abs(replacement_path.index(u) - replacement_path.index(v))==1:
      color = 'green'
    else:
      color = 'black'

    nt.add_edge(u, v, color = color, width = 5)

  nt.show('nx.html')


def generatePivots(g, TRAINING_SET_SIZE, visualize, pos):
    '''
    g: Networkx Graph
    TRAINING_SET_SIZE: Number of pivots

    For undirected, unweighted graphs. 
    Failing node > pivot node
    '''
    
    # Quick Sanity Check - Make Sure Connected and Undirected
    # for u in g.nodes:
    #     for v in g.nodes: 
    #       if not nx.has_path(g, u, v):
    #         continue
    #       assert(asp_length[u][v] == asp_length[v][u])
    # assert(nx.is_connected(g))

    # Replacement Paths    
    replacement_paths, replacement_paths_lengths = generateReplacementPaths(g, TRAINING_SET_SIZE, pos) 
    assert(TRAINING_SET_SIZE == len(replacement_paths_lengths))

    # Find Pivots
    pivots = {}
    for key_i, value_i in tqdm(replacement_paths.items(), desc = "Find Pivots in Replacement Paths", position = pos):
      source, target, failing_node = key_i

      assert(key_i not in pivots)
      pivots[key_i] = []
      
      # Look for Potential Pivots on the Replacement Path
      assert(type(value_i) is list)

      for replacement_path_j in value_i:
        assert(type(replacement_path_j) is list)
        for pivot_candidate in replacement_path_j[1:-1]:
          source_to_pivot = nx.shortest_path_length(g, source, pivot_candidate)
          pivot_to_target = nx.shortest_path_length(g, pivot_candidate, target)    

          if source_to_pivot + pivot_to_target == replacement_paths_lengths[key_i]:
            # print(pivot_candidate, [tmp for tmp in nx.all_shortest_paths(g, source, target)])
            # assert(replacement_path_j == nx.shortest_path(g, source, pivot_candidate)[:-1] + nx.shortest_path(g, pivot_candidate, target))
            pivots[key_i].append(pivot_candidate)
      if len(pivots[key_i])==0:
        print("NO PIVOTS FOUND")
        print(key_i, value_i)
        print(nx.shortest_path(g, source, pivot_candidate) + nx.shortest_path(g, pivot_candidate, target) )
        print(source_to_pivot + pivot_to_target, replacement_paths_lengths[key_i])
        pivots.pop(key_i)
      # assert(len(pivots[key_i])!=0)

    # Visualize: Deprecated for now
    # if visualize: 
    #     for key_i, pivot_node_list in random.sample(pivots.items(), 1):
    #       pivot_node = random.choice(pivot_node_list)
    #       source, target, failing_node = key_i  

    #       replacement_path_i = nx.shortest_path(g, source, pivot_node)[:-1] + nx.shortest_path(g, pivot_node, target)

    #       print("Example")
    #       print(f"Source: {source}, Target: {target}, Failing Node (Purple): {failing_node}, Pivots (Yellow): {pivot_node}")
    #       print(f"Original Shortest Path (Red): {asp_path[source][target]}, Length: {asp_length[source][target]}")
    #       print(f"Replacement Shortest Path (Green): {replacement_path_i}, Length: {replacement_paths_lengths[key_i]}")
    #       print()

    #       visualizeNodeRemoval(g, asp_path[source][target], failing_node, replacement_path_i, pivot = pivot_node)

    return list(pivots.items()) 