from .models.geom3d_models import Geom3D
from .models.CGCNN import CrystalGraphConvNet as CGCNN
from .models.simpleNet import SimpleNet
from .models.CrippenNet import CrippenNet, zinc_PARAMS
from .custom_sampler import SubsetWeightedRandomSampler
from .custom_dataset import QM9like

from torch_geometric.datasets import QM9
import torch
from torch import optim, nn
from torch.utils.data import Subset
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import curve_fit
import pickle
from tqdm import tqdm

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def prepare_data(data, N, extra_fea_matrix):
    """Create explicit adjacency matrix and feature matrix from single data point"""

    device = data.x.device
    
    n_onehot = len(extra_fea_matrix)
    
    atom_fea = torch.zeros(N, n_onehot+1, device=device)
    atom_fea[:data.x.shape[0],:n_onehot] = 1*data.x[:,:n_onehot]
    atom_fea[data.x.shape[0]:, n_onehot] = 1
    atom_fea = add_extra_features(atom_fea, extra_fea_matrix)
    atom_fea = atom_fea.unsqueeze(0)
    
    #N = data.x.shape[0]
    adj = torch.zeros(1,N,N,device=device)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[0,i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
        adj[0,j,i] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
    
    return atom_fea, adj

def prepare_data_vector(data, N, extra_fea_matrix, shuffle=False):
    """Create explicit adjacency matrix and feature matrix vector from mini-batch"""
        
    atom_fea = 1*data.x[:,:len(extra_fea_matrix)]
    
    atom_fea = add_extra_features(torch.cat([atom_fea, torch.zeros(atom_fea.shape[0],1, device=atom_fea.device)], dim=1), extra_fea_matrix)
        
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
        new_atom_fea[i,p[ten.shape[0]:],len(extra_fea_matrix)] = 1
        
    adj = torch.zeros(data.num_graphs, N,N, device=atom_fea.device)
    bond_type = torch.tensor([1,2,3,1.5], device=atom_fea.device)
    
    nn = data.batch[data.edge_index.T[:,0]]
    ij = data.edge_index.T - torch.cat([torch.zeros(1, dtype=torch.long, device=atom_fea.device), switch_points])[nn].unsqueeze(1).expand(data.edge_index.shape[1], 2)

    adj[nn,p[ij[:,0]], p[ij[:,1]]] = data.edge_attr.matmul(bond_type)
    
    return new_atom_fea, adj

def prepare_target_vector(data, N):
    """Same as prepare_data_vector for individual atomic contributions"""
    
    target = data.atom_class
    
    new_target = torch.zeros(data.num_graphs, N, target.shape[1], device=target.device)
        
    switch_points = torch.where(data.batch[1:] - data.batch[:-1] == 1)[0] + 1

    new_target_pieces = torch.tensor_split(target, switch_points.cpu())
    
    # This can be vectorized: https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    for i, ten in enumerate(new_target_pieces):
        new_target[i,:ten.shape[0],:] = ten
    
    return new_target

def class_stats(loader_train, show=False, device="cpu"):

    class_sum = []
    for batch_idx, data in enumerate(tqdm(loader_train)):
        data = data.to(device=device)
        class_sum.append(torch.sum(data.atom_class,dim=0).unsqueeze(0))

    class_sum = torch.cat(class_sum, dim=0)

    class_sum = torch.sum(class_sum, dim=0)

    print("Number of atom for each class:", class_sum)

    print("Most common class: %1.10f, %d"%((class_sum/torch.sum(class_sum)).max(), class_sum.max()), class_sum.argmax())
    print("Least common class: %1.10f, %d"%((class_sum/torch.sum(class_sum)).min(), class_sum.min()), class_sum.argmin())
    print("Least common non-zero class: %1.10f, %d"%((class_sum[class_sum > 0]/torch.sum(class_sum)).min(), class_sum[class_sum > 0].min()), np.arange(len(class_sum))[class_sum > 0][class_sum[class_sum > 0].argmin()])

    print("Total number of atoms:", torch.sum(class_sum))
    
    if show:
        plt.figure()

        plt.bar(torch.arange(len(class_sum)), class_sum, color="navy")

        ax = plt.gca()

        ax.set_yscale('log')

        plt.xticks(np.array([28, 41, 56, 57, 62, 65, 66, 67, 68, 69])-0.5)

        ax.tick_params(labelbottom=False)
        
        plt.show()

    weights = 1/(class_sum + 1)
    
    return weights


def add_extra_features(features, extra_fea_matrix):
    return torch.cat([features, torch.matmul(features[:,:len(extra_fea_matrix)], extra_fea_matrix)],dim=1)
                
def nudge(atom_fea, adj, noise_factor):
    atom_fea = atom_fea + torch.randn(*atom_fea.shape, device=atom_fea.device)*noise_factor
    adj = adj + torch.randn(*adj.shape, device=atom_fea.device)*noise_factor
    return atom_fea, adj

def shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def train(config, output):

    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize network

    if config.model == "CGCNN":
        model = CGCNN(config._extra_fea_matrix.shape[0] + config._extra_fea_matrix.shape[1] + 1,
                      atom_fea_len=config.atom_fea_len, n_conv=config.n_conv, h_fea_len=128, n_out=len(config.property)).to(device)
    elif config.model == "CrippenNet":
        model = CrippenNet(config._extra_fea_matrix.shape[0], n_conv=config.n_conv, layer_list=config.layer_list, classifier=config.atom_class).to(device)
    elif config.model in ["GIN", "GCN", "GAT", "GraphSAGE"]:
        model = Geom3D(config._extra_fea_matrix.shape[0] + config._extra_fea_matrix.shape[1] + 1, config.n_conv, config.atom_fea_len, JK=config.JK, drop_ratio=config.dropout, gnn_type=config.model, graph_pooling=config.pooling, num_tasks=len(config.property)).to(device)
    else:
        model = SimpleNet(config._extra_fea_matrix.shape[0] + config._extra_fea_matrix.shape[1] + 1, 
                          atom_fea_len=config.atom_fea_len, n_conv=config.n_conv, layer_list=config.layer_list, n_out=len(config.property),
                          pooling=config.pooling, dropout = config.dropout, batch_norm=config.batch_norm,
                          multiplier=config.output_multiplier).to(device)

    if config.use_pretrained or config.transfer_learn:
        if os.path.isfile(output+'/model_weights.pth'):
            model.load_state_dict(torch.load(output+'/model_weights.pth', map_location=device))
        else:
            raise RuntimeError("Trying to use pretrained model but %s/model_weights.pth does not exist"%(output))
    
    if not config.use_pretrained:
        
        def keep_in(data):
            return len(data.x) <= config.max_size
        
        dataset_dict = dict()

        for dset in config.datasets:
        
            if dset == "qm9":
                
                dataset_dict[dset] = QM9(output + "/" + dset, pre_filter=keep_in)
            
                if ["H", "C", "N", "O", "F"] != config.type_list:
                    raise RuntimeError("type_list is incompatible with dataset", list(dataset_dict["qm9"].types)) 
                
            else:
            
                dataset_dict[dset] = QM9like(output + "/" + dset, pre_filter=keep_in, type_list = config.type_list, atom_class=config.atom_class)
                dataset_dict[dset] = dataset_dict[dset][torch.randperm(len(dataset_dict[dset]))]
                                    
        all_train = torch.utils.data.ConcatDataset([data[:len(data)-len(data)//10] for data in dataset_dict.values()])
        all_valid = torch.utils.data.ConcatDataset([data[len(data)-len(data)//10:] for data in dataset_dict.values()])

        if config.n_data < len(all_train) + len(all_valid):
            
            all_train = Subset(all_train, torch.randperm(config.n_data - config.n_data//10))
            all_valid = Subset(all_valid, torch.randperm(config.n_data//10))
        
        print("Size of database:", len(all_train) + len(all_valid))
        
        loader_train = DataLoader(all_train, batch_size = config.batch_size, shuffle=True)
        loader_valid = DataLoader(all_valid, batch_size = config.batch_size, shuffle=True)
                
        # Loss and optimizer
        if config.atom_class:
            weights = class_stats(loader_train, config.show_train, device)
            criterion = nn.NLLLoss()

            if config.alpha > 0:
            
                cost_matrix = abs(torch.tensor(zinc_PARAMS, device=device).unsqueeze(1).expand(len(zinc_PARAMS), len(zinc_PARAMS)) - torch.tensor(zinc_PARAMS, device=device).unsqueeze(0).expand(len(zinc_PARAMS), len(zinc_PARAMS)))
            
                cost_matrix = cost_matrix/torch.max(cost_matrix)
            
        else:
            criterion = nn.L1Loss()

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        if config.show_train:
            # Initialize plot
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.plot([],'b',label = 'Train')
            ax1.plot([],'r',label = 'Validation')
            if config.atom_class:
                ax1.set_yscale("log")
            ax1.set_xlabel('Epoch')
            if config.atom_class:
                ax1.set_ylabel('Cross Entropy')
            else:
                ax1.set_ylabel('MAE (eV)')
            ax1.legend(loc=2)
        
        # Train Network
        epoch_loss_train = []
        epoch_loss_valid = []
        training_time = time.time()
        
        for epoch in range(config.num_epochs):
    
            model.train()

            if config.atom_class:
                if epoch != 0:
                    # Fraction of each class that is wrongly classified
                    perf = torch.sum(all_stats, dim=1)/(class_sum + 1e-12)
                    epoch_weights = weights*(1+2*perf)
                else:
                    epoch_weights = weights
            
            epoch_loss_train.append(0)
            epoch_scores = []
            epoch_targets = []
            correct = 0
            total = 0
            all_stats = torch.zeros(len(zinc_PARAMS), len(zinc_PARAMS), device=device)
            class_sum = torch.zeros(len(zinc_PARAMS), device=device)

            for batch_idx, data in enumerate(loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
                
                inputs = prepare_data_vector(data, config.max_size, config._extra_fea_matrix, shuffle=config.shuffle)
                
                inputs = nudge(*inputs, config.noise_factor) # Make the model more tolerent of non integers
                
                # forward
                torch.cuda.empty_cache()
                scores = model(*inputs)

                if config.atom_class:

                    target = prepare_target_vector(data, config.max_size)

                    target = target.view(-1, target.shape[2])
                    scores = scores.view(-1, scores.shape[2])

                    # Random undersampling
                    randomselect = torch.multinomial(target.matmul(epoch_weights), config.n_rand_samples, replacement=True)
                    
                    target = target[randomselect,:]
                    scores = scores[randomselect,:]

                    # Statistics
                    class_sum += torch.sum(target,dim=0)

                    scores_zeros = torch.sum(scores, dim=1)==0
                    target_zeros = torch.sum(target, dim=1)==0
                    
                    scores_max = torch.argmax(scores, dim=1)
                    target_max = torch.argmax(target, dim=1)
                    
                    correct += torch.sum((target_max == scores_max) | scores_zeros)
                    total += target.shape[0]
                    
                    should_be_idx = target_max[(target_max != scores_max) & (~scores_zeros)]
                    predicted_idx = scores_max[(target_max != scores_max) & (~scores_zeros)]

                    stats = torch.zeros((should_be_idx.shape[0], target.shape[1], target.shape[1]), device=device)
                    stats[torch.arange(should_be_idx.shape[0]), should_be_idx, predicted_idx] = 1
                    
                    stats = torch.sum(stats, dim=0)

                    all_stats += stats
                    
                    epoch_scores.append(torch.sum(scores.detach().matmul(torch.tensor(zinc_PARAMS, device=device))).unsqueeze(0))
                    epoch_targets.append(torch.sum(target.matmul(torch.tensor(zinc_PARAMS, device=device))).unsqueeze(0))

                    if config.alpha > 0:
                        loss = (1-config.alpha)*criterion(scores, torch.argmax(target, dim=1)) + config.alpha*torch.sum(target.matmul(cost_matrix)*scores)
                    else:
                        loss = criterion(scores, torch.argmax(target, dim=1))
                else:
                    target = data.y[:,config.property]
    
                    epoch_scores.append(scores.detach())
                    epoch_targets.append(target)

                    loss = criterion(scores, target)

                if config.atom_class:
                    epoch_loss_train[-1] += float(torch.mean(abs(epoch_scores[-1] - epoch_targets[-1])))
                else:
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
                correct_valid = 0
                total_valid = 0
                for batch_idx, data in enumerate(loader_valid):
                    # Get data to cuda if possible
                    data = data.to(device=device)
                    
                    # forward
                    scores_valid = model(*prepare_data_vector(data, config.max_size, config._extra_fea_matrix))

                    if config.atom_class:
                        target_valid = prepare_target_vector(data, config.max_size)

                        correct_valid += torch.sum(torch.argmax(target_valid, dim=2) == torch.argmax(scores_valid, dim=2))
                        total_valid += target_valid.shape[0]*target_valid.shape[1]
                        
                        epoch_scores_valid.append(torch.sum(scores_valid.matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1))
                        epoch_targets_valid.append(torch.sum(target_valid.matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1))

                        scores_valid = scores_valid.permute((0,2,1))
                        target_valid = target_valid.permute((0,2,1))
                    else:
                        target_valid = data.y[:,config.property]
                
                        epoch_scores_valid.append(scores_valid)
                        epoch_targets_valid.append(target_valid)
                    
                    loss = criterion(scores_valid, target_valid)

                    if config.atom_class:
                        epoch_loss_valid[-1] += float(torch.mean(abs(epoch_scores_valid[-1] - epoch_targets_valid[-1])))
                    else:
                        epoch_loss_valid[-1] += float(loss)
                
            epoch_loss_train[-1] = epoch_loss_train[-1]/len(loader_train)
            epoch_loss_valid[-1] = epoch_loss_valid[-1]/len(loader_valid)
            
            if config.atom_class:
                print(epoch, "AVG random TRAIN MAE", float(epoch_loss_train[-1]), "AVG VALID MAE", float(epoch_loss_valid[-1]), "TRAIN ACCURACY", correct/total,"VALID ACCURACY", correct_valid/total_valid, flush=True)
            else:
                print(epoch, "AVG TRAIN MAE", float(epoch_loss_train[-1]), "AVG VALID MAE", float(epoch_loss_valid[-1]), flush=True)
            
            if epoch%10 == 0 and config.show_train:
                ax2.clear()
                ax1.plot(epoch_loss_valid[:-1],'r',label = 'Validation')
                ax1.plot(epoch_loss_train[1:],'b',label = 'Train')
                ax2.plot(torch.cat(epoch_targets).cpu(), torch.cat(epoch_scores).cpu().detach().numpy(), ".", alpha=0.1)
                ax2.plot(torch.cat(epoch_targets_valid).cpu(), torch.cat(epoch_scores_valid).cpu().detach().numpy(), ".", alpha=0.1)
                x = np.linspace(0,18,300)
                ax2.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
                ax2.plot(x, x, color="k", alpha=0.5)
                ax2.set_aspect("equal","box")
                plt.pause(0.1)

        print("Total training time:", time.time() - training_time)
        
        torch.save(model.state_dict(), output+'/model_weights.pth')

        if config.show_train:
            plt.ioff()
            plt.savefig(output+"/progress.png", dpi=300)

        model.eval()
        with torch.no_grad():
            epoch_scores = []
            epoch_single_class_scores = []
            epoch_targets = []
            correct = 0
            total = 0
            all_stats = torch.zeros(len(zinc_PARAMS), len(zinc_PARAMS), device=device)
            class_sum = torch.zeros(len(zinc_PARAMS), device=device)
            for batch_idx, data in enumerate(loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
                
                inputs = prepare_data_vector(data, config.max_size, config._extra_fea_matrix, shuffle=False)
                
                if config.atom_class:

                    scores = model(*inputs)
                    target = prepare_target_vector(data, config.max_size)

                    class_sum += torch.sum(data.atom_class,dim=0)

                    scores_zeros = torch.sum(scores, dim=2)==0
                    target_zeros = torch.sum(target, dim=2)==0
                    
                    scores_max = torch.argmax(scores, dim=2)
                    target_max = torch.argmax(target, dim=2)
                    
                    correct += torch.sum((target_max == scores_max) | scores_zeros)
                    total += target.shape[0]*target.shape[1]
                    
                    should_be_idx = target_max[(target_max != scores_max) & (~scores_zeros)]
                    predicted_idx = scores_max[(target_max != scores_max) & (~scores_zeros)]

                    stats = torch.zeros((should_be_idx.shape[0], target.shape[2], target.shape[2]), device=device)
                    stats[torch.arange(should_be_idx.shape[0]), should_be_idx, predicted_idx] = 1

                    print("Stats shape", stats.shape)
                    
                    stats = torch.sum(stats, dim=0)

                    all_stats += stats
                    
                    single_class_scores = torch.sum(torch.tensor(zinc_PARAMS, device=device)[scores_max]*(~scores_zeros), dim=1) #TMP scores_max
                    scores = torch.sum(scores.matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1)
                    
                    target = torch.sum(prepare_target_vector(data, config.max_size).matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1)
                    
                    epoch_single_class_scores.append(single_class_scores)
                else:
                    scores = model(*inputs)
                    target = data.y[:,config.property]
                
                epoch_scores.append(scores)
                epoch_targets.append(target)
                            
            if config.num_epochs == 0:
                epoch_scores_valid = []
                epoch_targets_valid = []
                for batch_idx, data in enumerate(loader_valid):
                    # Get data to cuda if possible
                    data = data.to(device=device)
                
                    # forward
                    if config.atom_class:
                        scores_valid = torch.sum(model(*prepare_data_vector(data, config.max_size, config._extra_fea_matrix)).matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1)
                        target_valid = torch.sum(prepare_target_vector(data, config.max_size).matmul(torch.tensor(zinc_PARAMS, device=device)), dim=1)
                    else:
                        scores_valid = model(*prepare_data_vector(data, config.max_size, config._extra_fea_matrix))
                        target_valid = data.y[:,config.property]
            
                    epoch_scores_valid.append(scores_valid)
                    epoch_targets_valid.append(target_valid)

        plt.figure()

        train_targets = torch.cat(epoch_targets).cpu()
        train_scores = torch.cat(epoch_scores).cpu().detach().numpy()
        
        plt.plot(train_targets, train_scores, ".", alpha=0.1)

        if config.atom_class:
            
            print("Missclassification statistics per class (fraction wrong):")

            print(torch.sum(all_stats, dim=1)/class_sum)
        
            print("FINAL TRAIN ACCURACY", correct/total)
        
            print("FINAL TRAIN MAE", torch.mean(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))),
                  torch.mean(abs(torch.cat(epoch_targets) - torch.cat(epoch_single_class_scores))))

            print("FINAL TRAIN STDAE", torch.std(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))),
                  torch.std(abs(torch.cat(epoch_targets) - torch.cat(epoch_single_class_scores))))

            print("FINAL TRAIN maxAE", torch.max(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))),
                  torch.max(abs(torch.cat(epoch_targets) - torch.cat(epoch_single_class_scores))))

        else:    

            print("FINAL TRAIN MAE", torch.mean(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))))
            print("FINAL TRAIN STDAE", torch.std(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))))
            print("FINAL TRAIN maxAE", torch.max(abs(torch.cat(epoch_targets) - torch.cat(epoch_scores))))

            pickle.dump(all_stats, open(output+"/class_stats.pkl","wb"))
            
        valid_targets = torch.cat(epoch_targets_valid).cpu()
        valid_scores = torch.cat(epoch_scores_valid).cpu().detach().numpy()
        
        plt.plot(valid_targets, valid_scores,".", alpha=0.1)
        
        print("FINAL VALID MAE", torch.mean(abs(torch.cat(epoch_targets_valid) - torch.cat(epoch_scores_valid))))
        
        pickle.dump((train_targets, train_scores, valid_targets, valid_scores), open(output+"/final_performance_data.pkl","wb"))
        
        ax = plt.gca()
        
        x = np.linspace(0,18,300)
        ax.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
        
        plt.plot(x, x, color="k", alpha=0.5)
        plt.title("Property Model Performance")
        
        ax.set_aspect("equal","box")

        plt.savefig(output+"/final_performance.png", dpi=300)

        if config.show_train:
        
            plt.show()

    model.eval()
    
    return model

