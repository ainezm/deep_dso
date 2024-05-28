import torch
import numpy 
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F

class TinyModel(torch.nn.Module):

    def __init__(self, emb_list, edge_index, DROPOUT, device):
        super().__init__()
        
        self.number_of_nodes = len(emb_list)
        self.emb_size = len(emb_list[0])
        assert(self.emb_size == 128)

        self.initial_embeddings = torch.tensor(emb_list, requires_grad = False)
        self.edge_index = torch.tensor(edge_index, requires_grad = False)
        self.initial_embeddings, self.edge_index = self.initial_embeddings.to(device), self.edge_index.to(device)

        self.graph1 = GATConv(in_channels = self.emb_size, out_channels = 128, dropout = DROPOUT)
        self.graph1_gn = GraphNorm(in_channels = 128)
        self.graph2 = GATConv(in_channels = 128, out_channels = 128, dropout = DROPOUT)
        self.graph2_gn = GraphNorm(in_channels = 128)

        self.linear1 = torch.nn.Linear(128 * 3, 128 * 3)
        self.linear1_bn = torch.nn.BatchNorm1d(128*3)
        self.linear2 = torch.nn.Linear(128 * 3, self.number_of_nodes)

    def forward(self, x):
        mlp = self.graph1(x=self.initial_embeddings, edge_index=self.edge_index)
        mlp = F.relu(self.graph1_gn(mlp))
        mlp = self.graph2(x = mlp, edge_index = self.edge_index)
        mlp = F.relu(self.graph2_gn(mlp))

        if len(x.shape)==1: 
            mlp = self.linear1(torch.cat((torch.index_select(mlp, 0, x[0]), torch.index_select(mlp, 0, x[1]), torch.index_select(mlp, 0, x[2])), 1))
        elif len(x.shape)==2:
            mlp = self.linear1(torch.cat((torch.index_select(mlp, 0, x[:, 0]), torch.index_select(mlp, 0, x[:, 1]), torch.index_select(mlp, 0, x[:, 2])), 1))
        else:
            print("INCORRECT X SHAPE IN MODEL.PY")
        mlp = F.relu(self.linear1_bn(mlp))

        mlp = self.linear2(mlp)
        output = F.log_softmax(mlp)
        return output
        
