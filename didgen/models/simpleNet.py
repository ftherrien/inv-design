import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.sigmoid = nn.Sigmoid()
        self.sfotplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(self.atom_fea_len)
        self.linear = nn.Linear(self.atom_fea_len, self.atom_fea_len)
        

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
        
        normal_bonded_fea = self.bn1(bonded_fea)
        embed_normal_bond_fea = self.linear(normal_bonded_fea)
        out = self.sigmoid(embed_normal_bond_fea)
        #out = self.softplus(embed_normal_bond_fea)
        
        return out


class SimpleNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, size,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
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
        n_h: int
          Number of hidden layers after pooling
        """
        super(SimpleNet, self).__init__()
    
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.embedding2 = nn.Linear(atom_fea_len, h_fea_len)

        self.outlayer = nn.Linear(h_fea_len, 1)
        
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len)
                                    for _ in range(n_conv)])
        self.smart_pooling = nn.Linear(size, 1)

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
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, adj)

        mol_fea = self.smart_pooling(atom_fea.transpose(1,2)).squeeze() # Pooling: N0, N, F -> N0, F
            
        mol_fea = self.embedding2(mol_fea) 

        out = self.outlayer(self.sigmoid(mol_fea))

        return out
