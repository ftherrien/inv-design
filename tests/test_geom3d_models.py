import torch
import torch.nn as nn

# CH4
adj_CH4 = torch.tensor([[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]], dtype=torch.float)
edge_index_CH4 = torch.tensor([[0,1],[0,2],[0,3],[0,4],[1,0],[2,0],[3,0],[4,0]], dtype=torch.long).T
edge_attr_CH4 = torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1]], dtype=torch.float)
x_CH4 = torch.tensor([[0,1,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0]], dtype=torch.float)

# N2
adj_N2 = torch.tensor([[0,3,0,0,0],[3,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=torch.float)
edge_index_N2 = torch.tensor([[0,1],[1,0]], dtype=torch.long).T
edge_attr_N2 = torch.tensor([[3],[3]], dtype=torch.float)
x_N2 = torch.tensor([[0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=torch.float)

def test_GINConv():
    from didgen.models.geom3d_models import GINConv
    from didgen.models.originals.geom3d_models import GINConv as GINConv_orig

    emb_dim = 10

    conv = GINConv(emb_dim, seed=0)
    conv_orig = GINConv_orig(emb_dim, seed=0)
    fea_emb = nn.Linear(5, emb_dim)
    
    out = conv(fea_emb(x_N2), adj_N2)
    out_orig = conv_orig(fea_emb(x_N2), edge_index_N2, edge_attr_N2)

    assert (out == out_orig).all(), "Failed to match the original implementation with N2"

    out = conv(fea_emb(x_CH4), adj_CH4)
    out_orig = conv_orig(fea_emb(x_CH4), edge_index_CH4, edge_attr_CH4)

    assert (out == out_orig).all(), "Failed to match the original implementation with CH4"

def test_GCNConv():
    from didgen.models.geom3d_models import GCNConv
    from didgen.models.originals.geom3d_models import GCNConv as GCNConv_orig

    emb_dim = 10

    conv = GCNConv(emb_dim, seed=0)
    conv_orig = GCNConv_orig(emb_dim, seed=0)
    fea_emb = nn.Linear(5, emb_dim)
    
    out = conv(fea_emb(x_N2), adj_N2)
    out_orig = conv_orig(fea_emb(x_N2), edge_index_N2, edge_attr_N2)
    
    assert (abs(out - out_orig) < 1e-5).all(), "Failed to match the original implementation with N2"

    out = conv(fea_emb(x_CH4), adj_CH4)
    out_orig = conv_orig(fea_emb(x_CH4), edge_index_CH4, edge_attr_CH4)

    assert (abs(out - out_orig) < 1e-5).all(), "Failed to match the original implementation with CH4"

def test_Geom3D():
    from didgen.models.geom3d_models import Geom3D

    adj = torch.stack([adj_N2, adj_CH4])
    fea = torch.stack([x_N2, x_CH4])
    
    model = Geom3D(5, 3, 10, JK="last", drop_ratio=0, gnn_type="GIN", graph_pooling="mean", num_tasks=1)
    out = model(fea, adj)

    assert out.shape == (2, 1), "Failed to get the correct output shape with GIN"
    
    model = Geom3D(5, 3, 10, JK="last", drop_ratio=0, gnn_type="GCN", graph_pooling="mean", num_tasks=1)
    out = model(fea, adj)

    assert out.shape == (2, 1), "Failed to get the correct output shape with GCN"
    
if __name__ == "__main__":
    test_GINConv()
    

    
