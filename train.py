# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import optim, nn  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!
# from torch.utils.data import DataLoader
import numpy as np

# From Alex
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.Draw import MolToImage
import rdkit.Chem as Chem

# From me
from torch_geometric.nn import GCNConv
from model_vector import CrystalGraphConvNet as CGCNN
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time

## HYPERPARAMETERS =================

n_data = 690
num_epochs = 1000
batch_size = 230
orig_atom_fea_len = 5
nbr_fea_len = 4
learning_rate = 0.001
max_size = 10
use_pretrained = True

# ==================================

def MolFromGraph(features, adjacency_matrix):

    atoms = np.array(["H","C","N","O","F"])
    
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(features)):
        a = Chem.Atom(atoms[(features[i,:]==1).numpy()][0])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol

def prepare_data(data):
    # Get to correct shape
    atom_fea = data.x[:,:orig_atom_fea_len]
    N = len(data.x)
    adj = torch.zeros(N,N)
    nbr_fea = torch.zeros(N, N, nbr_fea_len)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[i,j] = 1
        adj[j,i] = 1
        nbr_fea[i,j,:] = data.edge_attr[n,:]
        nbr_fea[j,i,:] = data.edge_attr[n,:]
    
    crystal_atom_idx = []
    for i in range(data.num_graphs):
        crystal_atom_idx.append(torch.where(data.batch==i)[0])
    
    return atom_fea, nbr_fea, adj, crystal_atom_idx

def prepare_data_vector(data):
    # Get to correct shape
    atom_fea = data.x[:,:orig_atom_fea_len]

    if orig_atom_fea_len >= 12:
        data.x[:,5] = data.x[:,5]/9
    
    atom_fea = torch.cat([atom_fea, torch.matmul(atom_fea[:,:5], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1)
    
    N = torch.max(torch.tensor([torch.sum(data.batch==i) for i in range(data.num_graphs)]))
    
    crystal_atom_idx = []
    new_atom_fea = torch.zeros(data.num_graphs, N, atom_fea.shape[1])
    for i in range(data.num_graphs):
        crystal_atom_idx.append(torch.where(data.batch==i)[0])
        new_atom_fea[i,:len(crystal_atom_idx[-1]),:] = atom_fea[crystal_atom_idx[-1],:]
    
    adj = torch.zeros(data.num_graphs, N,N)
    nbr_fea = torch.zeros(data.num_graphs, N, N, nbr_fea_len)
    for n,(i,j) in enumerate(data.edge_index.T):

        nn = data.batch[i]
        ii = i - crystal_atom_idx[nn][0]
        jj = j - crystal_atom_idx[nn][0]
        
        adj[nn,ii,jj] = 1
        nbr_fea[nn,ii,jj,:] = data.edge_attr[n,:]
    
    return new_atom_fea, nbr_fea, adj

def prepare_data_from_features(features, adj):
    # Get to correct shape

    N = features.shape[0]
 
    new_atom_fea = torch.cat([features, torch.matmul(features[:,:5], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1).unsqueeze(0)

    adj = adj.unsqueeze(0)
    
    nbr_fea = torch.cat([adj.unsqueeze(3), torch.zeros((1, N, N, nbr_fea_len-1))],dim=3)
    
    return new_atom_fea, nbr_fea, adj
                
# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def keep_in(data):
    return len(data.x) <= max_size

# Load Training and Test data
qm9 = datasets.QM9("qm9_small", pre_filter=keep_in)

print("Size of database:", len(qm9))

qm9 = qm9[torch.randperm(len(qm9))]

print("QM9 EXAMPLE", qm9[0], qm9[0].x, qm9[0].edge_attr)

qm9_loader_train = DataLoader(qm9[:n_data], batch_size = batch_size, shuffle = True)
qm9_loader_valid = DataLoader(qm9[n_data:n_data + n_data//10], batch_size = batch_size, shuffle = True)

std = torch.std(qm9.data.y[:n_data,4])
mean = torch.mean(qm9.data.y[:n_data,4])

print("MEAN GAP:", mean)

# Initialize network
model = CGCNN(orig_atom_fea_len+1, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False).to(device)

if use_pretrained:
    model.load_state_dict(torch.load('model_weights.pth'))
else:
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
    
    # Initialize plot
    plt.ion()
    plt.figure()
    plt.axhline(std, color = 'k', label = 'STD')
    plt.plot([],'b',label = 'Train')
    plt.plot([],'r',label = 'Validation')
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (eV)')
    plt.legend(loc=2)
    
    # Train Network
    epoch_loss_train = []
    epoch_loss_valid = []
    for epoch in range(num_epochs):
        epoch_loss_train.append(0)
        for batch_idx, data in enumerate(qm9_loader_train):
            # Get data to cuda if possible
            data = data.to(device=device)
            
    
            inputs = prepare_data_vector(data)
    
            
            # forward
            scores = model(*inputs).squeeze()
            
         
            loss = criterion(scores, data.y[:,3]-data.y[:,2])
            epoch_loss_train[-1] += float(loss)
    
                   
            # backward
            optimizer.zero_grad()
            loss.backward()
    
            # gradient descent or adam step
            optimizer.step()
    
        epoch_loss_valid.append(0)
        for batch_idx, data in enumerate(qm9_loader_valid):
            # Get data to cuda if possible
            data = data.to(device=device)
    
            # forward
            scores = model(*prepare_data_vector(data)).squeeze()
            loss = criterion(scores, data.y[:,3]-data.y[:,2])
    
            epoch_loss_valid[-1] += float(loss)
            
        epoch_loss_train[-1] = epoch_loss_train[-1]/len(qm9_loader_train)
        epoch_loss_valid[-1] = epoch_loss_valid[-1]/len(qm9_loader_valid)
        print("AVG TRAIN RMSE", epoch_loss_train[-1], "AVG VALID RMSE", epoch_loss_valid[-1])
        #scheduler.step(epoch_loss_train[-1])
        
        if epoch%10 == 0:
            plt.plot(epoch_loss_train[1:],'b',label = 'Train')
            plt.plot(epoch_loss_valid[:-1],'r',label = 'Validation')
            plt.pause(0.1)
    
    torch.save(model.state_dict(), 'model_weights.pth')
            
    plt.ioff()
    plt.savefig("progress.png")

# Inverter -----------------------------------------------------------------------------

N = 15

sig = nn.Sigmoid()

sig_strength = 1

# Random Iinit
#adj_sqrt = torch.randn((1, N, N), requires_grad=True)
adj_vec = torch.randn((N*(N-1)//2), requires_grad=True)

fea_h = torch.randn((1, N, orig_atom_fea_len), requires_grad=True)

tidx = torch.tril_indices(row=N, col=N, offset=-1)

def to_model_inputs(adj_sqrt, fea_h):
    
    #atom_fea = sig(sig_strength*fea_h)
    atom_fea = fea_h
    
    atom_fea = torch.nn.functional.normalize(atom_fea,p=1, dim=2)

    atom_fea_ext = torch.cat([atom_fea, torch.matmul(atom_fea[0,:,:5], torch.tensor([[[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]]))],dim=2)

    #adj = adj_sqrt.matmul(adj_sqrt.transpose(1,2))

    adj = torch.zeros((N,N))
    
    adj[tidx[0],tidx[1]] = adj_vec

    adj = adj + adj.transpose(0,1)
    
    adj = sig(sig_strength*adj.unsqueeze(0))
    #adj = adj.unsqueeze(0)
    
    nbr_fea = torch.cat([adj.unsqueeze(3), torch.zeros((1, N, N, nbr_fea_len-1))],dim=3)

    return atom_fea_ext, nbr_fea, adj

for param in model.parameters():
    param.requires_grad = False

criterion = nn.MSELoss()
optimizer = optim.SGD([adj_vec, fea_h], lr=0.1)

plt.ion()
plt.figure()
plt.plot([],'b')
plt.xlabel('Epoch')
plt.ylabel('Difference (eV)')
plt.ylim([0,5])

diffs = []
for e in range(10000):
    score = model(*to_model_inputs(adj_vec, fea_h))
        
    loss = abs(score - torch.tensor([[1.0]]))
    diffs.append(float(loss)) 
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # gradient descent or adam step
    optimizer.step()

    if e%10 == 0:
        print(loss, score)
    
    if e%1000 == 0:
        plt.plot(diffs,'b')
        plt.pause(0.1)

plt.ioff()
        
atom_fea_ext, _, adj = to_model_inputs(adj_vec, fea_h)

print(atom_fea_ext, adj)

print("Final value:", model(*to_model_inputs(adj_vec, fea_h)))

idx = torch.argmax(atom_fea_ext[0,:,:5], dim=1)

features = torch.zeros((N,5))

for i,j in enumerate(idx):
    features[i,j] = 1

adj = torch.round(adj).squeeze()

print(features, adj)

print("Final value after rounding:", model(*prepare_data_from_features(features, adj)))

mol = MolFromGraph(features, adj)

img = MolToImage(mol)

img.save("generated_mol.png")

plt.figure()

data_train = next(iter(DataLoader(qm9[:n_data], batch_size = n_data, shuffle = False)))
plt.plot(data_train.y[:,4], model(*prepare_data_vector(data_train)).detach().numpy(), ".")

data_valid = next(iter(DataLoader(qm9[n_data:n_data + n_data//10], batch_size = n_data//10, shuffle = False)))
plt.plot(data_valid.y[:,4], model(*prepare_data_vector(data_valid)).detach().numpy(),".")

plt.show()
