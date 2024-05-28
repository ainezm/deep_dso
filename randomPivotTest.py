import os
import networkx as nx
from pyvis import network as net
import random
import math 
import numpy as np
from tqdm import tqdm 
import torch
import time
import datetime 
import multiprocessing

# Generating Graph
from graphs.graph import readNetworkRepositoryGraph
log_file = open("log.txt", "a")

for GRAPH_NAME in ['bn-mouse-kasthuri_graph_v4.edges']:
    print(GRAPH_NAME)
    
    g = readNetworkRepositoryGraph(absFilePath = os.path.abspath(f"./graphs/network_repository/{GRAPH_NAME}"))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    print(GRAPH_NAME, "NODES:", g.number_of_nodes(), "DENSITY:", nx.density(g), "AVERAGE DEGREE:", 2*g.number_of_edges()/g.number_of_nodes())
    g = nx.convert_node_labels_to_integers(g, label_attribute=None).copy()

    # Generating Pivots 
    from graphs import pivot_node
    TRAINING_SET_SIZE = min(int(g.number_of_nodes() ** 2), 10 ** 5)
    TRAINING_SET_SIZE //= 10
    print("TRAINING_SET_SIZE:", TRAINING_SET_SIZE)

    n_cpu = multiprocessing.cpu_count()
    print(f"CPU COUNT: {n_cpu}")

    pivots_arguments = [(g, TRAINING_SET_SIZE//n_cpu, False, k) for k in range(n_cpu)]
    pivots = []
    with multiprocessing.Pool(processes=n_cpu, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        res = pool.starmap(pivot_node.generatePivots, pivots_arguments)
        for k in res:
            pivots += k 
    assert(type(pivots)==list and len(pivots) == TRAINING_SET_SIZE)
    
    # # Generating Dataset
    from dataset import prepare_data
    test_dl = prepare_data(pivots, g.number_of_nodes(), batch_size = 256, mode = 'test')

    # Test Script
    from metrics.stretch import evaluateStretch
    from models.model import TinyModel
    
    test_stretch = []
    for t_inputs, t_targets in test_dl:
        t_pred = np.random.uniform(size = t_targets.shape)
        test_stretch += evaluateStretch(g, t_inputs.numpy(), t_pred, t_targets.numpy())

    log_file.write(f"{GRAPH_NAME} {sum(test_stretch)/len(test_stretch)}\n")
    log_file.flush()
    print(GRAPH_NAME, sum(test_stretch)/len(test_stretch))

log_file.close()