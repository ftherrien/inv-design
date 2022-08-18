# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import optim, nn  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!
# from torch.utils.data import DataLoader

# From Alex
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
# from rdkit.Chem.rdmolfiles import SDMolSupplier

# From me
from torch_geometric.nn import GCNConv
from model_vector import CrystalGraphConvNet as CGCNN
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time

## HYPERPARAMETERS =================

n_data = 100
num_epochs = 2000
batch_size = 100
orig_atom_fea_len = 5
nbr_fea_len = 4
learning_rate = 0.001

# ==================================

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

    N = torch.max(torch.tensor([torch.sum(data.batch==i) for i in range(data.num_graphs)]))
    
    crystal_atom_idx = []
    new_atom_fea = torch.zeros(data.num_graphs, N, orig_atom_fea_len)
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
                
# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Training and Test data
qm9 = datasets.QM9("qm9")
qm9_loader_train = DataLoader(qm9[:n_data], batch_size = batch_size, shuffle = True)
qm9_loader_valid = DataLoader(qm9[n_data:n_data + n_data//10], batch_size = batch_size, shuffle = False)


std = torch.std(qm9[:n_data].data.y[:,4])
mean = torch.mean(qm9[:n_data].data.y[:,4])

print("MEAN GAP:", mean)

# Initialize network
model = CGCNN(orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
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
        plt.plot(epoch_loss_valid[1:],'r',label = 'Validation')
        plt.pause(0.1)

torch.save(model.state_dict(), 'model_weights.pth')
        
plt.ioff()
plt.savefig("progress.png")
plt.show()
