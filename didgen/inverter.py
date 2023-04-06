from .utils import draw_mol, round_mol, MolFromGraph
from .train import prepare_data

import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit.Chem.Draw import MolToImage
import os

sig = nn.Sigmoid()
soft = nn.Softmax(dim=2)
soft0 = nn.Softmax(dim=0)

def start_from(data, config):

    device = data.x.device
    
    N = config.max_size

    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    # Get to correct shape
    atom_fea = torch.zeros(N, config.n_onehot+1, device=device)
    atom_fea[:data.x.shape[0],:config.n_onehot] = 1*data.x[:,:config.n_onehot]
    atom_fea[data.x.shape[0]:, config.n_onehot] = 1
    
    atom_fea = atom_fea.unsqueeze(0)
    
    #N = atom_fea.shape[1]
    
    adj = torch.zeros(N, N, device=device)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))

    adj_vec = torch.zeros(N*(N-1)//2, device=device)
    for i, t in enumerate(tidx.T):
        adj_vec[i] = adj[t[0],t[1]]
        
    adj_vec = torch.sqrt(adj_vec)
        
    return atom_fea, adj_vec

def smooth_round(x):
    return (396*torch.pi*x - 225*torch.sin(2*torch.pi*x) + 45*torch.sin(4*torch.pi*x) - 5*torch.sin(6*torch.pi*x))/(396*torch.pi)
    # return torch.round(x) + 0.1*(x - torch.round(x))

def max_round(x):

    idx = torch.argmax(x[0,:,:], dim=1)
    
    n_x = 0.3*x
    
    for i,j in enumerate(idx):
        n_x[0,i,j] = 1 - 0.3*(1-x[0,i,j])
    return n_x

    # return x

def weights_to_model_inputs(fea_h, adj_vec, config):

    device = adj_vec.device
    
    N = fea_h.shape[1]
    
    tidx = torch.tril_indices(row=N, col=N, offset=-1)

    atom_fea = fea_h**2/torch.sum(fea_h**2,dim=2,keepdim=True).expand((1, fea_h.shape[1], fea_h.shape[2]))
    
    # atom_fea = soft(config.sig_strength*(fea_norm-0.5))
    
    mr_atom_fea = max_round(atom_fea)
    
    atom_fea_ext = torch.cat([mr_atom_fea, torch.matmul(mr_atom_fea[0,:,:config.n_onehot], torch.tensor([[[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]], device = device))],dim=2)
    
    adj = torch.zeros((N,N), device=device)
    
    adj[tidx[0],tidx[1]] = adj_vec**2
    
    adj = adj + adj.transpose(0,1)

    adj = smooth_round(adj)
    
    r_features, r_adj = round_mol(atom_fea_ext, adj, config.n_onehot)

    bonds_per_atom = torch.matmul(r_features[:,:config.n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0], device = device))

    constraints = torch.sum((torch.sum(adj, dim=1) - bonds_per_atom)**2)
    
    integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:config.n_onehot+1])**2)

    integer_adj = torch.sum((r_adj - adj)**2)

    # integer_fea = torch.sum(torch.sin(atom_fea_ext[0,:,:config.n_onehot+1]*torch.pi)**6)

    # integer_adj = torch.sum(torch.sin(adj*torch.pi)**6)
    
    adj = adj.unsqueeze(0)

    return atom_fea_ext, adj, constraints, r_features, r_adj

def gauss(x, p):
    return torch.exp(-(x - p)**2/0.25)

def get_connected(adj, idx, i):
    for j in torch.nonzero(adj[i,:]):
        j = int(j)
        if j not in idx:
            idx.append(j)
            idx = get_connected(adj,idx,j)
    return idx

def get_components(adj):
    idx = []
    allidx = set()
    while allidx != set(range(adj.shape[0])):
        rest = set(range(adj.shape[0])) - allidx
        el = rest.pop()
        idx.append(get_connected(adj,[el],el))
        allidx = set().union(*idx)
    return idx


def initialize(qm9, config, output, i, device="cpu"):

    N = config.max_size

    if type(config.starting_size) == list:
        max_N = int(torch.randint(*config.starting_size, (1,)))
    else:
        max_N = config.starting_size
        
    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    if config.start_from == "random":

        os.makedirs(output + "/save_random", exist_ok=True)
        
        adj_vec = torch.randn((N*(N-1)//2), device=device)
        
        fea_h = torch.randn((1, N, config.n_onehot+1), device=device)

        pickle.dump((adj_vec, fea_h), open(output + "/save_random/%d.pkl"%(i),"wb")) # Save the random full matrices 
        
        # Shorten the input to fit max_N 

        adj_vec[(tidx[0] > max_N - 1) | (tidx[1] > max_N - 1)] = 0
        
        fea_h[0,max_N:,:] = 0

        fea_h[0,max_N:,config.n_onehot] = 1
        
    elif config.start_from == "saved":
        adj_vec, fea_h = pickle.load(open(output + "/save_random/%d.pkl"%(i),"rb"))

        # Shorten the input to fit max_N 

        adj_vec[(tidx[0] > max_N - 1) | (tidx[1] > max_N - 1)] = 0
        
        fea_h[0,max_N:,:] = 0

        fea_h[0,max_N:,config.n_onehot] = 1
        
    else:
        qmid = int(config.start_from)
        fea_h, adj_vec = start_from(qm9[qmid].to(device), config)
    
        init_features, init_adj = prepare_data(qm9[qmid].to(device), config.max_size, config.n_onehot)
    
        mol = MolFromGraph(init_features[0,:,:config.n_onehot],init_adj.squeeze(), config.n_onehot)
    
        img = MolToImage(mol)
    
        img.save(output+"/initial_mol.png")
    
        
    #print("INIT ADJ_VEC", adj_vec)
    #print("INIT FEA_H", fea_h)
    
    adj_vec.requires_grad = True
    fea_h.requires_grad = True

    return fea_h, adj_vec

def invert(target_value, model, fea_h, adj_vec, config, output):
    """ Backpropagates all the way to the input to produce a molecule with a specific property """

    # Freezes the trained model
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
    # optimizer = optim.SGD([adj_vec, fea_h], lr=config.inv_r)

    if config.show_losses:
        plt.ion()
        plt.figure()
        plt.plot([],'b')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Difference (eV)')
    
    losses = []
    total_losses = []
    constraint_losses = []
    integer_fea_losses = []
    integer_adj_losses = []
    g1s = []
    g2s = []
    extra = 0
    cycle = False
    dip = 0
    ccount = 0
    lcount = 0

    if config.show_losses:
        loop = range(config.n_iter)
    else:
        loop = tqdm(range(config.n_iter))
    
    for e in loop:
        
        atom_fea_ext, adj, constraints, r_features, r_adj = weights_to_model_inputs(fea_h, adj_vec, config)
        
        score = model(atom_fea_ext, adj)
        
        loss = abs(score - torch.tensor([[target_value]], device=score.device))
        
        integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:config.n_onehot+1])**2)

        integer_adj = torch.sum((r_adj - adj.squeeze())**2)
        
        r_bonds_per_atom = torch.matmul(r_features, torch.tensor([1.0,4.0,3.0,2.0,1.0,0.0], device=score.device))
        
        if torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12 and loss < config.stop_loss:

            # Number of components in graph
            L = torch.diag(torch.sum((r_adj != 0),axis=0)) - (r_adj != 0)*1
            n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - int(torch.sum(r_features[:,config.n_onehot]))

            if n_comp == 1:
                break
            else:

                #draw_mol(atom_fea_ext, adj, config.n_onehot, output)
                
                comps = get_components(r_adj)
                idx = comps[torch.argmax(torch.tensor([len(c) for c in comps]))]

                if len(idx) >= config.min_size:
                
                    tidx = torch.tril_indices(row=r_adj.shape[0], col=r_adj.shape[0], offset=-1)
                    
                    with torch.no_grad():
                        
                        adj_vec[torch.all(tidx[0].unsqueeze(1).expand(-1,len(idx)) != torch.tensor([idx]).expand(len(tidx[0]),-1), dim=1) & torch.any(tidx[1].unsqueeze(1).expand(-1,len(idx)) != torch.tensor([idx]).expand(len(tidx[1]),-1), dim=1)] = 0
                        
                        new_fea_h = 0*fea_h
                        
                        new_fea_h[0,:,config.n_onehot] = 1
                    
                        new_fea_h[0,idx,:] = fea_h[0,idx,:]
                    
                        fea_h = new_fea_h
                        
                    # atom_fea_ext, adj, constraints, r_features, r_adj = weights_to_model_inputs(fea_h, adj_vec, config)
                    
                    # draw_mol(atom_fea_ext, adj, config.n_onehot, output)
                                        
                    adj_vec, fea_h = adj_vec.detach(), fea_h.detach()
                    
                    adj_vec.requires_grad = True
                    fea_h.requires_grad = True
                    
                    optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
                    
                    continue

        total_loss = (config.l_loss*loss**2 + config.l_const*constraints**2)/(config.l_loss*loss + config.l_const*constraints)

        total_losses.append(float(total_loss))
        losses.append(float(loss))
        constraint_losses.append(float(constraints))
        integer_fea_losses.append(float(integer_fea))
        integer_adj_losses.append(float(integer_adj))
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
        
        if e%10 == 0 and config.show_losses:
            #print(c[0], c[1], n_dip, float(loss), torch.max(abs(fea_h)), torch.max(abs(adj_vec)))
            print(float(total_loss), float(loss), float(constraints), float(integer_fea), float(integer_adj), float(score))
            
        if e%1000 == 0 and config.show_losses:
            plt.plot(total_losses,'k')
            plt.plot(losses,'b')
            plt.plot(constraint_losses,".-", color="r")
            plt.plot(integer_fea_losses, color="orange")
            plt.plot(integer_adj_losses, color="purple")
            plt.plot(g1s, color="pink")
            plt.plot(g2s, color="green")
            plt.pause(0.1)
            draw_mol(atom_fea_ext, adj, config.n_onehot, output, text="%f"%(score), color=(int(255*(float(loss/score))),int(255*(1-float(loss/score))),0))

    if config.show_losses:
        plt.ioff()

    # print(fea_h, adj_vec)
        
    return fea_h, adj_vec, e
