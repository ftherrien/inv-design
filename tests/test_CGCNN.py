import torch
import torch.nn as nn

# CH4
adj_CH4 = torch.tensor([[0,1,1,1,1,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]], dtype=torch.float)
x_CH4 = torch.tensor([[0,1,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]], dtype=torch.float)
nbr_idx_CH4 = torch.ones((6,6), dtype=torch.long)*5
nbr_idx_CH4[0,1:5] = torch.arange(1,5)
nbr_idx_CH4[1:5,0] = 0

# N2
adj_N2 = torch.tensor([[0,3,0,0,0,0],[3,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]], dtype=torch.float)
edge_index_N2 = torch.tensor([[0,1],[1,0]], dtype=torch.long).T
x_N2 = torch.tensor([[0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],[0,0,0,0,0]], dtype=torch.float)
nbr_idx_N2 = torch.ones((6,6), dtype=torch.long)*5
nbr_idx_N2[0,1] = 1
nbr_idx_N2[1,0] = 0

def test_ConvLayer():
    from didgen.models.CGCNN import ConvLayer
    from didgen.models.originals.CGCNN import ConvLayer as ConvLayer_orig

    emb_dim = 10

    conv = ConvLayer(emb_dim, seed=0)
    conv_orig = ConvLayer_orig(emb_dim,1, seed=0)
    fea_emb = nn.Linear(5, emb_dim, bias=False)
    
    out = conv(fea_emb(x_N2).unsqueeze(0), adj_N2.unsqueeze(0))
    out_orig = conv_orig(fea_emb(x_N2), adj_N2.unsqueeze(-1), nbr_idx_N2)
    
    assert (abs(out - out_orig) < 1e-5).all(), "Failed to match the original implementation with N2"

    out = conv(fea_emb(x_CH4).unsqueeze(0), adj_CH4.unsqueeze(0))
    out_orig = conv_orig(fea_emb(x_CH4), adj_CH4.unsqueeze(-1), nbr_idx_CH4)

    assert (abs(out - out_orig) < 1e-5).all(), "Failed to match the original implementation with CH4"

def test_CrystalGraphConvNet():
    from didgen.models.CGCNN import CrystalGraphConvNet

    adj = torch.stack([adj_N2, adj_CH4])
    fea = torch.stack([x_N2, x_CH4])
    
    model = CrystalGraphConvNet(5, atom_fea_len=10, n_conv=3, h_fea_len=128, n_h=1, classification=False, n_out=1)
    out = model(fea, adj)

    assert out.shape == (2, 1), "Failed to get the correct output shape with GIN"

    
if __name__ == "__main__":
    test_ConvLayer()
    test_CrystalGraphConvNet()
    

    
