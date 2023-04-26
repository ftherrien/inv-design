from .models.CGCNN import CrystalGraphConvNet as CGCNN
from .custom_sampler import SubsetWeightedRandomSampler

import torch
from torch import optim, nn
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import curve_fit
import pickle

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def prepare_data(data, N, n_onehot):
    """Create explicit adjacency matrix and feature matrix from single data point"""

    device = data.x.device
    
    atom_fea = torch.zeros(N, n_onehot+1, device=device)
    atom_fea[:data.x.shape[0],:n_onehot] = 1*data.x[:,:n_onehot]
    atom_fea[data.x.shape[0]:, n_onehot] = 1
    atom_fea = torch.cat([atom_fea, torch.matmul(atom_fea[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]], device=device))],dim=1)
    atom_fea = atom_fea.unsqueeze(0)
    
    #N = data.x.shape[0]
    adj = torch.zeros(1,N,N,device=device)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[0,i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
        adj[0,j,i] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
    
    return atom_fea, adj

def prepare_data_vector(data, N, n_onehot, shuffle=False):
    """Create explicit adjacency matrix and feature matrix vector from mini-batch"""

    t = time.time()
        
    atom_fea = 1*data.x[:,:n_onehot]

    atom_fea = torch.cat([atom_fea, torch.zeros(atom_fea.shape[0],1, device=atom_fea.device), torch.matmul(atom_fea[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]], device=atom_fea.device))],dim=1)
    
    #N = torch.max(torch.tensor([torch.sum(data.batch==i) for i in range(data.num_graphs)]))
    
    new_atom_fea = torch.zeros(data.num_graphs, N, atom_fea.shape[1], device=atom_fea.device)
        
    switch_points = torch.where(data.batch[1:] - data.batch[:-1] == 1)[0] + 1

    new_atom_fea_pieces = torch.tensor_split(atom_fea, switch_points.cpu())

    if shuffle:
        p = torch.randperm(N, device=atom_fea.device)
    else:
        p = torch.arange(N, device=atom_fea.device)
    
    # This can be vectorized: https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    for i, ten in enumerate(new_atom_fea_pieces):
        new_atom_fea[i,p[:ten.shape[0]],:] = ten
        new_atom_fea[i,p[ten.shape[0]:],n_onehot] = 1
        
    adj = torch.zeros(data.num_graphs, N,N, device=atom_fea.device)
    bond_type = torch.tensor([1,2,3,1.5], device=atom_fea.device)
    
    nn = data.batch[data.edge_index.T[:,0]]
    ij = data.edge_index.T - torch.cat([torch.zeros(1, dtype=torch.long, device=atom_fea.device), switch_points])[nn].unsqueeze(1).expand(data.edge_index.shape[1], 2)

    adj[nn,p[ij[:,0]], p[ij[:,1]]] = data.edge_attr.matmul(bond_type)
    
    return new_atom_fea, adj

def prepare_data_from_features(features, adj, n_onehot):
    """From simple adj and feature matrix add atoms info and adjust dimensions from model input"""

    N = features.shape[0]
 
    new_atom_fea = torch.cat([features, torch.matmul(features[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]], device=features.device))],dim=1).unsqueeze(0)

    adj = adj.unsqueeze(0)
    
    return new_atom_fea, adj
                
def nudge(atom_fea, adj, noise_factor):
    atom_fea = atom_fea + torch.randn(*atom_fea.shape, device=atom_fea.device)*noise_factor
    adj = adj + torch.randn(*adj.shape, device=atom_fea.device)*noise_factor
    return atom_fea, adj

def shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def get_samplers(qm9, config):

    y, bins = torch.histogram(qm9.data.y[:,4], 1000)

    bins[0] -= 1e-12
    bins[-1] += 1e-12

    bin_idx = torch.bucketize(qm9.data.y[:,4], bins) - 1

    x = (bins[1:] + bins[:-1])/2

    p, pcov = curve_fit(gauss, x.detach().numpy(), y.detach().numpy())

    normal_y = gauss(x,*p)

    print("Closest gaussian distribution: A: %f, mu: %f, sigma:%f"%tuple(p))
    
    threshold = 100
    
    bin_weights = torch.ones(len(y))
    bin_weights[(normal_y > threshold) & (normal_y > threshold)] = gauss(x[(normal_y > threshold) & (normal_y > threshold)],*p)/y[(normal_y > threshold) & (normal_y > threshold)]

    pickle.dump(bin_weights, open("weights.pkl", "wb"))
    
    weights = bin_weights[bin_idx]

    valid_size = config.n_data//10
    train_size = config.n_data - valid_size
    
    # This will determine the valid-train split
    if config.random_split:
        idx = torch.randint(0, len(qm9), (train_size + valid_size,))
    else:
        idx = list(range(len(qm9)))
    
    return SubsetWeightedRandomSampler(weights, idx[:train_size]), SubsetWeightedRandomSampler(weights, idx[train_size:train_size+valid_size])

def train(qm9, config, output):
    """ Training of CGCNN (vectorized) on the qm9 database """

    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize network

    if config.model == "SimpleNet":
        model = SimpleNet(config.n_onehot+2, config.max_size, 
                          atom_fea_len=64, n_conv=3, h_fea_len=config.max_size, n_h=1,
                          classification=False).to(device)
    else:
        model = CGCNN(config.n_onehot+2,
                      atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                      classification=False).to(device)

    if config.use_pretrained or config.transfer_learn:
        if os.path.isfile(output+'/model_weights.pth'):
            model.load_state_dict(torch.load(output+'/model_weights.pth', map_location=device))
        else:
            raise RuntimeError("Trying to use pretrained model but %s/model_weights.pth does not exist"%(output))
    
    if not config.use_pretrained:
        
        print("Size of database:", len(qm9))

        train_sampler, valid_sampler = get_samplers(qm9, config)

        qm9_loader_train = DataLoader(qm9, batch_size = config.batch_size, sampler=train_sampler)
        qm9_loader_valid = DataLoader(qm9, batch_size = config.batch_size, sampler=valid_sampler)
    
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
        training_time = time.time()
        for epoch in range(config.num_epochs):
    
            model.train()
            
            epoch_loss_train.append(0)
            epoch_scores = []
            epoch_targets = []
            for batch_idx, data in enumerate(qm9_loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
                
                inputs = prepare_data_vector(data, config.max_size, config.n_onehot, shuffle=True)
    
                inputs = nudge(*inputs, config.noise_factor) # Make the model more tolerent of non integers
                
                # forward
                scores = model(*inputs).squeeze()
                
                target = data.y[:,3]-data.y[:,2]
    
                epoch_scores.append(scores.detach())
                epoch_targets.append(target)
                
                loss = criterion(scores, target)
                epoch_loss_train[-1] += float(loss)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
        
                # gradient descent or adam step
                optimizer.step()
    
            model.eval()
            with torch.no_grad():
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
            print(epoch, "AVG TRAIN RMSE", float(epoch_loss_train[-1]), "AVG VALID RMSE", float(epoch_loss_valid[-1]), flush=True)
            #scheduler.step(epoch_loss_train[-1])
            
            if epoch%10 == 0:
                ax2.clear()
                ax1.plot(epoch_loss_train[1:],'b',label = 'Train')
                ax1.plot(epoch_loss_valid[:-1],'r',label = 'Validation')
                ax2.plot(torch.cat(epoch_targets).cpu(), torch.cat(epoch_scores).cpu().detach().numpy(), ".", alpha=0.1)
                ax2.plot(torch.cat(epoch_targets_valid).cpu(), torch.cat(epoch_scores_valid).cpu().detach().numpy(), ".", alpha=0.1)
                x = np.linspace(0,18,300)
                ax2.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
                ax2.plot(x, x, color="k", alpha=0.5)
                ax2.set_aspect("equal","box")
                plt.pause(0.1)

        print("Total training time:", time.time() - training_time)
        
        torch.save(model.state_dict(), output+'/model_weights.pth')
        
        plt.ioff()
        plt.savefig(output+"/progress.png")

        model.eval()
        with torch.no_grad():
            epoch_scores = []
            epoch_targets = []
            for batch_idx, data in enumerate(qm9_loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
                
                inputs = prepare_data_vector(data, config.max_size, config.n_onehot, shuffle=True)
                            
                scores = model(*inputs).squeeze()
                target = data.y[:,3]-data.y[:,2]
                
                epoch_scores.append(scores)
                epoch_targets.append(target)
                            
            if config.num_epochs == 0:
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

        plt.figure()


        train_targets = torch.cat(epoch_targets).cpu()
        train_scores = torch.cat(epoch_scores).cpu().detach().numpy()
        
        plt.plot(train_targets, train_scores, ".", alpha=0.1)
        
        print("FINAL TRAIN RMSE", criterion(torch.cat(epoch_targets), torch.cat(epoch_scores)))

        valid_targets = torch.cat(epoch_targets_valid).cpu()
        valid_scores = torch.cat(epoch_scores_valid).cpu().detach().numpy()
        
        plt.plot(valid_targets, valid_scores,".", alpha=0.1)
        
        print("FINAL TEST RMSE", criterion(torch.cat(epoch_targets_valid), torch.cat(epoch_scores_valid)))

        pickle.dump((train_targets, train_scores, valid_targets, valid_scores), open(output+"/final_performance_data.pkl","wb"))
        
        ax = plt.gca()
        
        x = np.linspace(0,18,300)
        ax.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
        
        plt.plot(x, x, color="k", alpha=0.5)
        plt.title("Property Model Performance")
        
        ax.set_aspect("equal","box")

        plt.savefig(output+"/final_performance.png")

    model.eval()
        
    return model

