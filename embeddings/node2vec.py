
from ge import Node2Vec
import tensorflow as tf

def get_embeddings(g, walk_length, num_walks,p , q, workers, window_size, iter):    
    model = Node2Vec(g, walk_length = walk_length, num_walks = num_walks,p = p, q = q, workers = workers)#init model
    model.train(window_size = window_size, iter = iter)# train model
    node2vec_list = model.get_embeddings()# get embedding vectors

    return node2vec_list