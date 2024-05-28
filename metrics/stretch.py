import networkx as nx
from pyvis import network as net
import random
import math 
import numpy as np
from tqdm import tqdm 

'''
We define stretch as: 
ratio = (dist(s, p, G-e) + dist(p, t, G-e)) / dist(s, t, G-e)
'''
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


def evaluateStretch(g, inputs, pred, targets):
  # print(pred, targets)
  assert(len(inputs)==len(pred) and len(pred)==len(targets))
  assert(len(inputs.shape)==2 and len(pred.shape)==2 and len(targets.shape)==2)

  original_g = g.copy()
  stretch_list = []

  for input_i, pred_list, targets_i in zip(inputs, pred, np.argmax(targets, axis = 1)):
    s_i, t_i, f_i = input_i
    # print(f"s_i {s_i}, t_i {t_i}, f_i {f_i}, targets_i {targets_i}")
    
    original_edges = [k for k in original_g.edges(f_i)]
    original_g.remove_node(f_i)
  
    if nx.shortest_path_length(original_g, s_i, t_i) != nx.shortest_path_length(original_g, s_i, targets_i) + nx.shortest_path_length(original_g, targets_i, t_i):
      print("STRETCH ERROR", s_i, t_i, f_i, (nx.shortest_path_length(original_g, s_i, t_i), nx.shortest_path_length(original_g, s_i, targets_i) + nx.shortest_path_length(original_g, targets_i, t_i)))
      visualizeNodeRemoval(g, list(nx.shortest_path(original_g, s_i, t_i)), list(nx.shortest_path(original_g, s_i, targets_i) + nx.shortest_path(original_g, targets_i, t_i)))
    assert(nx.shortest_path_length(original_g, s_i, t_i) == nx.shortest_path_length(original_g, s_i, targets_i) + nx.shortest_path_length(original_g, targets_i, t_i))

    for pred_i in np.argsort(pred_list)[::-1]:
      # print("pred_i", pred_i)
      if original_g.has_node(pred_i) and nx.has_path(original_g, s_i, pred_i) and nx.has_path(original_g, pred_i, t_i):
        stretch_list.append((nx.shortest_path_length(original_g, s_i, pred_i) + nx.shortest_path_length(original_g, pred_i, t_i))/nx.shortest_path_length(original_g, s_i, t_i))

        break

    original_g.add_edges_from(original_edges) 
  
  return stretch_list
