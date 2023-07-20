import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, dropout=None):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nonlinear = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(self.atom_fea_len)
        self.linear = nn.Linear(self.atom_fea_len, self.atom_fea_len)

        if dropout is not None:
            self.dplayer = nn.Dropout(dropout)
            self.dropout = True
        else:
            self.dropout = False

    def forward(self, atom_in_fea, adj):
        """
        Forward pass

        N: Total number of atoms in the structure
        N0: Total number of structures per batch

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N0, N, atom_fea_len)
          Atom hidden features before convolution
        adj: torch.LongTensor shape (N0, N, N)
          adjacency matrix

        Returns
        -------

        atom_out_fea: nn.Variable shape (N0, N, atom_fea_len)
          Atom hidden features after convolution

        """
        N0 = atom_in_fea.shape[0]
        N = atom_in_fea.shape[1]
        
        bonded_fea = adj.matmul(atom_in_fea) # (N0, N, atom_fea_len)
        normal_bonded_fea = self.bn(bonded_fea.permute((1,2,0))).permute(2,0,1)
        if self.dropout:
            normal_bonded_fea = self.dplayer(normal_bonded_fea)
        embed_normal_bond_fea = self.linear(normal_bonded_fea)
        out = self.nonlinear(embed_normal_bond_fea)
        
        return out


class SimpleNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len,
                 atom_fea_len=64, n_conv=3, layer_list=[128], n_out=1,
                 classification=False, pooling="sum", dropout=None):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_out: int
          Number of output layers
        """
        super(SimpleNet, self).__init__()
    
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.pooling_weights = nn.Linear(atom_fea_len, 1)
        
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, dropout=dropout)
                                    for _ in range(n_conv)])

        self.nonlinear = nn.Softplus()

        self.pooling = pooling

        if dropout is None:
            seq = (nn.Linear(atom_fea_len, layer_list[0]), self.nonlinear)
            for i in range(len(layer_list)-1):
                seq += (nn.Linear(layer_list[i], layer_list[i+1]), self.nonlinear)
            seq += (nn.Linear(layer_list[-1], 1), self.nonlinear)
        else:
            seq = (nn.Dropout(dropout), nn.Linear(atom_fea_len, layer_list[0]), self.nonlinear)
            for i in range(len(layer_list)-1):
                seq += (nn.Dropout(dropout), nn.Linear(layer_list[i], layer_list[i+1]), self.nonlinear)
            seq += (nn.Dropout(dropout), nn.Linear(layer_list[-1], n_out), self.nonlinear)
             
        self.deep = nn.Sequential(*seq)
        
    def forward(self, atom_fea, adj):
        """
        Forward pass

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N0, N, orig_atom_fea_len)
          Atom features from atom type
        adj: torch.LongTensor shape (N0, N, N)
          adjacency matrix

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        conv_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            conv_fea = conv_func(conv_fea, adj)

        if self.pooling == "sum":

            mol_fea = torch.sum(conv_fea, dim=1).squeeze(dim=1)

        elif self.pooling == "smarter":
            
            pooling_weights = self.pooling_weights(conv_fea).transpose(1,2) # N0, N, F -> N0, 1, N
            
            mol_fea = pooling_weights.matmul(conv_fea).squeeze(dim=1) # pooling N0, N, F -> N0, F

        elif self.pooling == "smartest":
            
            pooling_weights = self.nonlinear(self.pooling_weights(conv_fea).transpose(1,2)) # N0, N, F -> N0, 1, N
            
            mol_fea = pooling_weights.matmul(conv_fea).squeeze(dim=1) # pooling N0, N, F -> N0, F

        return self.deep(mol_fea)
