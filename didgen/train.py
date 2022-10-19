from .models.CGCNN import CrystalGraphConvNet as CGCNN

import torch
from torch import optim, nn
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os

def prepare_data(data, N, n_onehot):
    """Create explicit adjacency matrix and feature matrix from single data point"""

    atom_fea = torch.zeros(N, n_onehot+1)
    atom_fea[:data.x.shape[0],:n_onehot] = 1*data.x[:,:n_onehot]
    atom_fea[data.x.shape[0]:, n_onehot] = 1
    atom_fea = torch.cat([atom_fea, torch.matmul(atom_fea[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1)
    atom_fea = atom_fea.unsqueeze(0)
    
    #N = data.x.shape[0]
    adj = torch.zeros(1,N,N)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[0,i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5]))
        adj[0,j,i] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5]))
    
    return atom_fea, adj

def prepare_data_vector(data, N, n_onehot):
    """Create explicit adjacency matrix and feature matrix vector from mini-batch"""
    
    atom_fea = 1*data.x[:,:n_onehot]

    atom_fea = torch.cat([atom_fea, torch.zeros(atom_fea.shape[0],1), torch.matmul(atom_fea[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1)
    
    #N = torch.max(torch.tensor([torch.sum(data.batch==i) for i in range(data.num_graphs)]))
    
    crystal_atom_idx = []
    new_atom_fea = torch.zeros(data.num_graphs, N, atom_fea.shape[1])
    for i in range(data.num_graphs):
        crystal_atom_idx.append(torch.where(data.batch==i)[0])
        new_atom_fea[i,:len(crystal_atom_idx[-1]),:] = atom_fea[crystal_atom_idx[-1],:]
        new_atom_fea[i,len(crystal_atom_idx[-1]):,n_onehot] = 1
    
    adj = torch.zeros(data.num_graphs, N,N)
    for n,(i,j) in enumerate(data.edge_index.T):

        nn = data.batch[i]
        ii = i - crystal_atom_idx[nn][0]
        jj = j - crystal_atom_idx[nn][0]
        
        adj[nn,ii,jj] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5]))
        
    return new_atom_fea, adj

def prepare_data_from_features(features, adj, n_onehot):
    """From simple adj and feature matrix add atoms info and adjust dimensions from model input"""

    N = features.shape[0]
 
    new_atom_fea = torch.cat([features, torch.matmul(features[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1).unsqueeze(0)

    adj = adj.unsqueeze(0)
    
    return new_atom_fea, adj
                
def nudge(atom_fea, adj, noise_factor):
    atom_fea = atom_fea + torch.randn(*atom_fea.shape)*noise_factor
    adj = adj + torch.randn(*adj.shape)*noise_factor
    return atom_fea, adj

def shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def train(qm9, config, output):
    """ Training of CGCNN (vectorized) on the qm9 database """

    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize network
    model = CGCNN(config.n_onehot+2,
                     atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                     classification=False).to(device)
    
    if config.use_pretrained and os.path.isfile(output+'/model_weights.pth'):
        model.load_state_dict(torch.load(output+'/model_weights.pth'))
    else:
        print("Size of database:", len(qm9))
    
        qm9_loader_train = DataLoader(qm9[:config.n_data], batch_size = config.batch_size, shuffle = True)
        qm9_loader_valid = DataLoader(qm9[config.n_data:config.n_data + config.n_data//10], batch_size = config.batch_size, shuffle = True)
    
        std = torch.std(qm9.data.y[:config.n_data,4])
        mean = torch.mean(qm9.data.y[:config.n_data,4])
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        
        # Initialize plot
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.axhline(std, color = 'k', label = 'STD')
        ax1.plot([],'b',label = 'Train')
        ax1.plot([],'r',label = 'Validation')
        ax1.set_yscale("log")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE (eV)')
        ax1.legend(loc=2)
        
        # Train Network
        epoch_loss_train = []
        epoch_loss_valid = []
        for epoch in range(config.num_epochs):
    
            model.train()
            
            epoch_loss_train.append(0)
            epoch_scores = []
            epoch_targets = []
            for batch_idx, data in enumerate(qm9_loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
        
                inputs = prepare_data_vector(data, config.max_size, config.n_onehot)
    
                inputs = nudge(*inputs, config.noise_factor) # Make the model more tolerent of non integers
                
                # forward
                scores = model(*inputs).squeeze()
                target = data.y[:,3]-data.y[:,2]
    
                epoch_scores.append(scores)
                epoch_targets.append(target)
                
                loss = criterion(scores, target)
                epoch_loss_train[-1] += float(loss)
        
                       
                # backward
                optimizer.zero_grad()
                loss.backward()
        
                # gradient descent or adam step
                optimizer.step()
    
            model.eval()
                
            epoch_loss_valid.append(0)
            epoch_scores_valid = []
            epoch_targets_valid = []
            for batch_idx, data in enumerate(qm9_loader_valid):
                # Get data to cuda if possible
                data = data.to(device=device)
                
                # forward
                scores_valid = model(*prepare_data_vector(data, config.max_size, config.n_onehot)).squeeze()
                target_valid = data.y[:,3]-data.y[:,2]
    
                epoch_scores_valid.append(scores_valid)
                epoch_targets_valid.append(target_valid)
                
                loss = criterion(scores_valid, target_valid)
                epoch_loss_valid[-1] += float(loss)
                
            epoch_loss_train[-1] = epoch_loss_train[-1]/len(qm9_loader_train)
            epoch_loss_valid[-1] = epoch_loss_valid[-1]/len(qm9_loader_valid)
            print(epoch, "AVG TRAIN RMSE", float(epoch_loss_train[-1]), "AVG VALID RMSE", float(epoch_loss_valid[-1]))
            #scheduler.step(epoch_loss_train[-1])
            
            if epoch%10 == 0:
                ax2.clear()
                ax1.plot(epoch_loss_train[1:],'b',label = 'Train')
                ax1.plot(epoch_loss_valid[:-1],'r',label = 'Validation')
                ax2.plot(torch.cat(epoch_targets), torch.cat(epoch_scores).detach().numpy(), ".")
                ax2.plot(torch.cat(epoch_targets_valid), torch.cat(epoch_scores_valid).detach().numpy(), ".")
                x = np.linspace(0,18,300)
                ax2.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
                ax2.plot(x, x, color="k", alpha=0.5)
                ax2.set_aspect("equal","box")
                plt.pause(0.1)

        torch.save(model.state_dict(), output+'/model_weights.pth')

        model.eval()
        
        plt.ioff()
        plt.savefig("progress.png")

        plt.figure()
        
        criterion = nn.MSELoss()
        
        data_train = next(iter(DataLoader(qm9[:config.n_data], batch_size = config.n_data, shuffle = False)))
        plt.plot(data_train.y[:,4], model(*prepare_data_vector(data_train, config.max_size, config.n_onehot)).detach().numpy(), ".")
        
        print("FINAL TRAIN RMSE", criterion(data_train.y[:,4], model(*prepare_data_vector(data_train, config.max_size, config.n_onehot)).squeeze()))
        
        data_valid = next(iter(DataLoader(qm9[config.n_data:config.n_data + config.n_data//10], batch_size = config.n_data//10, shuffle = False)))
        plt.plot(data_valid.y[:,4], model(*prepare_data_vector(data_valid, config.max_size, config.n_onehot)).detach().numpy(),".")
        
        print("FINAL TEST RMSE", criterion(data_valid.y[:,4], model(*prepare_data_vector(data_valid, config.max_size, config.n_onehot)).squeeze()))
        
        ax = plt.gca()
        
        x = np.linspace(0,18,300)
        ax.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
        
        plt.plot(x, x, color="k", alpha=0.5)
        plt.title("Property Model Performance")
        
        ax.set_aspect("equal","box")

    model.eval()
        
    return model

