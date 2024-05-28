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
from graphs.graph import readNetworkRepositoryGraph, generateGeometricGraph
log_file = open("log.txt", "a")

for GRAPH_NAME in ['bn-mouse-kasthuri_graph_v4.edges']:
    print(GRAPH_NAME)

    BEST_STRETCH = 1e10
    BEST_LR = -1
    BEST_DROPOUT = -1
    
    g = readNetworkRepositoryGraph(absFilePath = os.path.abspath(f"./graphs/network_repository/{GRAPH_NAME}"))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    print(GRAPH_NAME, "NODES:", g.number_of_nodes(), "DENSITY:", nx.density(g), "AVERAGE DEGREE:", 2*g.number_of_edges()/g.number_of_nodes())
    g = nx.convert_node_labels_to_integers(g, label_attribute=None).copy()
    
    # Generating Pivots 
    from graphs import pivot_node
    TRAINING_SET_SIZE = min(int(g.number_of_nodes() ** 2), 10 ** 5)
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

    # Generate Initial Node2Vec Embeddings
    from embeddings import node2vec
    start_time = time.time()
    emb_list = node2vec.get_embeddings(g, walk_length = 10, num_walks = 80, p = 1, q = 1, workers = 4, window_size = 10, iter = 16) 
    end_time = time.time()
    print("Elapsed Time:", str(datetime.timedelta(seconds=end_time-start_time)))
    assert(len(emb_list) == g.number_of_nodes())
    emb_list = [np.array(emb_list[i]) for i in range(0, len(emb_list))]
    emb_list = np.stack(emb_list, axis = 0)
    
    # # Generating Dataset
    from dataset import prepare_data
    train_dl, val_dl, test_dl = prepare_data(pivots, g.number_of_nodes(), batch_size = 256)

    for DROPOUT in [0.10]: #0.10, 0.30, 0.50
        for LEARNING_RATE in [0.0010]: #0.0001, 0.0005, 0.0010     
            # Define the Model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Torch is using device {device}")

            from models.model import TinyModel
            edge_list = [list(k) for k in g.edges()]
            edge_list = np.transpose(np.array(edge_list))
            model = TinyModel(emb_list, edge_list, DROPOUT = DROPOUT, device = device)
            print(model)
            model.to(device)

            # Train the Model

            from torch.optim import Adam
            from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
            criterion = BCEWithLogitsLoss()
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

            from torch.utils.tensorboard import SummaryWriter
            import tensorflow as tf
            import tensorboard
            tf.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
            tb = SummaryWriter(f'runs/{GRAPH_NAME}_experiment_0_GATConv_lr{LEARNING_RATE}_dropout{DROPOUT}_multilabel')

            from metrics.stretch import evaluateStretch

            LOCAL_STRETCH = 1e10
            for epoch in tqdm(range(32),desc = "Training"):
                # enumerate mini batches
                train_loss = 0.0
                train_stretch = []
                model.train()
                for i, (inputs, targets) in enumerate(train_dl):
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    # for name, para in model.named_parameters():
                    #     print(f"{name} {para}")
                    # print(model.linear1.weight.grad)
                    # print(model.graph1.lin.weight.grad)

                    pred = model(inputs)
                    loss = criterion(pred, targets)
                    # if epoch %100==0:
                    #     print(f"epoch {epoch} pred {torch.argmax(pred)} actual {torch.argmax(targets)} loss {loss}")
                    loss.backward()
                    # for name, para in model.named_parameters():
                    #     print(f"{name} {para}")
                    # print(model.linear1.weight.grad)
                    # print(model.graph1.lin.weight.grad)

                    optimizer.step()

                    train_loss += loss.item() * inputs.shape[0]
                    train_stretch += evaluateStretch(g, inputs.cpu().detach().numpy(), pred.cpu().detach().numpy(), targets.cpu().detach().numpy())
                train_loss /= len(train_dl.dataset)

                valid_loss = 0.0
                valid_stretch = []
                model.eval()
                for v_inputs, v_targets in val_dl:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_pred = model(v_inputs) 
                    loss = criterion(v_pred, v_targets)
                    valid_loss += loss.item() * v_inputs.shape[0]
                    valid_stretch += evaluateStretch(g, v_inputs.cpu().detach().numpy(), v_pred.cpu().detach().numpy(), v_targets.cpu().detach().numpy())
                valid_loss /= len(val_dl.dataset)

                tb.add_scalars('Loss',
                        { 'Training' : train_loss, 'Validation' : valid_loss },
                        epoch + 1)
                tb.add_scalars('Stretch',
                        { 'Training' : sum(train_stretch)/len(train_stretch), 'Validation' : sum(valid_stretch)/len(valid_stretch) },
                        epoch + 1)
                tb.flush()

                LOCAL_STRETCH = min(LOCAL_STRETCH, sum(valid_stretch)/len(valid_stretch))
                if sum(valid_stretch)/len(valid_stretch) < BEST_STRETCH:
                    BEST_STRETCH = sum(valid_stretch)/len(valid_stretch)
                    BEST_DROPOUT = DROPOUT
                    BEST_LR = LEARNING_RATE
                    torch.save(model.state_dict(), f"bestmodel_{GRAPH_NAME}.pth")
            log_file.write(f"Network: {GRAPH_NAME} LR: {LEARNING_RATE} DP: {DROPOUT} STRETCH: {LOCAL_STRETCH}\n")
            # tb.add_graph(model = model, input_to_model = inputs[0])
            # tb.add_embedding(mat  = emb_list)
            tb.close()

    # Define the Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Torch is testing using device {device}")

    from models.model import TinyModel
    edge_list = [list(k) for k in g.edges()]
    edge_list = np.transpose(np.array(edge_list))
    model = TinyModel(emb_list, edge_list, DROPOUT = BEST_DROPOUT, device = device)
    model.to(device)
    model.load_state_dict(torch.load(f"bestmodel_{GRAPH_NAME}.pth"))

    test_stretch = []
    model.eval()
    for t_inputs, t_targets in test_dl:
        t_inputs, t_targets = t_inputs.to(device), t_targets.to(device)
        t_pred = model(t_inputs) 
        loss = criterion(t_pred, t_targets)
        test_stretch += evaluateStretch(g, t_inputs.cpu().detach().numpy(), t_pred.cpu().detach().numpy(), t_targets.cpu().detach().numpy())

    log_file.write(f"BEST: {GRAPH_NAME} {sum(test_stretch)/len(test_stretch)} LR:{BEST_LR} DROPOUT:{BEST_DROPOUT}\n")
    log_file.flush()
    print(GRAPH_NAME, sum(test_stretch)/len(test_stretch), f"LR:{BEST_LR} DROPOUT:{BEST_DROPOUT}")

log_file.close()
