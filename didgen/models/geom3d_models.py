import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GlobalAttention, MessagePassing, Set2Set,
                                global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
# from ogb.graphproppred.mol_encoder import full_atom_feature_dims , full_bond_feature_dims 

class permutedBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, x):
        if x.dim() == 3:
            return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return super().forward(x)

def propagate(self, x, adj, **kwargs):

    adj_ones = adj.clone()
    adj_ones[adj_ones > 1] = 1
    
    #adj_ones.matmul(x)
    #matmul = (adj.unsqueeze(-1) * x.unsqueeze(-3)).sum(-2)

    x_j = adj_ones.unsqueeze(-1) * x.unsqueeze(-3) # x_j-ish, includes edges
    
    before_agg = self.message(x_j, adj_ones=adj_ones, x_i=x.unsqueeze(-2).expand(x_j.shape) ,**kwargs)
    
    return self.update(before_agg.sum(-2), **kwargs)

def custom_softmax(src, mask, dim=-1):
    mask = mask.expand(src.shape)
    out = src.exp() * mask**2 / (src.exp() * mask).sum(dim, keepdim=True)
    out[mask == 0] = 0
    return out
    
class GINConv(MessagePassing):
    def __init__(self, emb_dim, seed=None):

        if seed is not None:
            torch.manual_seed(seed)
        
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), permutedBatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = nn.Linear(1, emb_dim, bias=False) #self.bond_encoder = nn.Embedding(full_bond_feature_dims[0], emb_dim)

    def forward(self, x, adj):

        # adj (N0, N, N)
        # x (N0, N, fea_len)

        edge_embedding = self.bond_encoder(adj.unsqueeze(-1))

        out = self.mlp((1 + self.eps) *x + propagate(self, x, adj, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr, **kwargs):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out, **kwargs):
        return aggr_out
    
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, seed=None):
        super(GCNConv, self).__init__(aggr='add')

        if seed is not None:
            torch.manual_seed(seed)
        
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = nn.Linear(1, emb_dim, bias=False)

    def forward(self, x, adj):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(adj.unsqueeze(-1))

        adj_ones = adj.clone()
        adj_ones[adj_ones > 1] = 1

        row = adj_ones.sum(-1, keepdim=True) + 1
        col = adj_ones.sum(-2, keepdim=True) + 1
        
        norm = 1/torch.sqrt(row * col).unsqueeze(-1)
        
        return propagate(self, x, adj, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) / row

    def message(self, x_j, edge_attr, norm, **kwargs):
        return norm * F.relu(x_j + edge_attr)

    def update(self, aggr_out, **kwargs):
        return aggr_out

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", seed=None):
        super(GATConv, self).__init__(node_dim=0)

        if seed is not None:
            torch.manual_seed(seed)
        
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.bond_encoder = nn.Linear(1, heads * emb_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, adj):
        edge_embedding = self.bond_encoder(adj.unsqueeze(-1))

        x = self.weight_linear(x)
        return propagate(self, x, adj, edge_attr=edge_embedding)

    def message(self, x_j, x_i, adj_ones, edge_attr):
        x_i = x_i.view(*x_i.shape[:-1], self.heads, self.emb_dim)
        x_j = x_j.view(*x_j.shape[:-1], self.heads, self.emb_dim)
        edge_attr = edge_attr.view(*edge_attr.shape[:-1], self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att.view(*(1,)*(len(x_j.shape)-2), *self.att.shape)).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = custom_softmax(alpha, adj_ones.unsqueeze(-1), dim=-3)

        return (x_j * alpha.unsqueeze(-1)).movedim(-2,-3)
        
    def update(self, aggr_out, **kwargs):
        aggr_out = aggr_out.mean(dim=-2)
        aggr_out += self.bias
        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean", seed=None):
        super(GraphSAGEConv, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
        
        self.emb_dim = emb_dim
        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), permutedBatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.bond_encoder = nn.Linear(1, emb_dim, bias=False)
        self.aggr = aggr

    def forward(self, x, adj):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(adj.unsqueeze(-1))

        return propagate(self, x, adj, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr, **kwargs):
        return x_j + edge_attr

    def update(self, aggr_out, **kwargs):
        return F.normalize(aggr_out, p=2, dim=-1)  
    
class GNN(nn.Module):
    def __init__(self, orig_emb_dim, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="GIN"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = nn.Linear(orig_emb_dim, emb_dim) #self.atom_encoder = nn.Embedding(full_atom_feature_dims[0], emb_dim)
        
        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "GIN":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "GCN":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "GAT":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "GraphSAGE":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise ValueError("Undefined GNN type")

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(permutedBatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, fea, adj):

        x = self.atom_encoder(fea)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], adj)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

class Geom3D(nn.Module):
    def __init__(self, orig_emb_dim, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="GIN", graph_pooling="mean", num_tasks=1):
        super(Geom3D, self).__init__()

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.molecule_model = GNN(orig_emb_dim, num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = lambda x: torch.sum(x, dim=1).squeeze(dim=1)
        elif graph_pooling == "mean":
            self.pool = lambda x: torch.mean(x, dim=1).squeeze(dim=1)
        elif graph_pooling == "max":
            self.pool = lambda x: torch.max(x, dim=1).squeeze(dim=1)
        else:
            raise ValueError("Invalid graph pooling type.")

        if JK == "concat":
            self.graph_pred_linear = nn.Linear((num_layer + 1) * emb_dim, num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(emb_dim, num_tasks)
        return

    def forward(self, x, adj):

        node_representation = self.molecule_model(x, adj)
        graph_representation = self.pool(node_representation)

        output = self.graph_pred_linear(graph_representation)

        return output
   
