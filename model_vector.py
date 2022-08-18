from __future__ import print_function, division

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, adj, nbr_fea):
        """
        Forward pass

        N: Total number of atoms in the structure
        N0: Total number of structures per batch

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N0, N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N0, N, N, nbr_fea_len)
          Bond features of each atom's M neighbors
        adj: torch.LongTensor shape (N0, N, N)
          adjacency matrix

        Returns
        -------

        atom_out_fea: nn.Variable shape (N0, N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N0 = atom_in_fea.shape[0]
        N = atom_in_fea.shape[1]
        
        # convolution
        ADJ = adj.unsqueeze(3).expand(N0, N, N, self.atom_fea_len)
        FEA = atom_in_fea.unsqueeze(2).expand(N0, N, N, self.atom_fea_len)
        atom_nbr_fea = ADJ * FEA
        total_nbr_fea = torch.cat(
            [FEA,
             atom_nbr_fea, nbr_fea], dim=3)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            N0, -1, self.atom_fea_len*2).permute((1,2,0))).permute((2,0,1)).view(N0, N, N, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=3)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=2)
        nbr_sumed = self.bn2(nbr_sumed.permute((1,2,0))).permute((2,0,1))
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, adj):
        """
        Forward pass

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N0, N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N0, N, N, nbr_fea_len)
          Bond features of each atom's N neighbors
        adj: torch.LongTensor shape (N0, N, N)
          adjacency matrix

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, adj, nbr_fea)
        crys_fea = torch.mean(atom_fea,1)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    # def pooling(self, atom_fea):
    #     """
    #     Pooling the atom features to crystal features

    #     N: Total number of atoms in the batch
    #     N0: Total number of crystals in the batch

    #     Parameters
    #     ----------

    #     atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
    #       Atom feature vectors of the batch
    #     crystal_atom_idx: list of torch.LongTensor of length N0
    #       Mapping from the crystal idx to atom idx
    #     """
    #     return torch.mean(atom_fea, dim=0, keepdim=True)
