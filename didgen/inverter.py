from .utils import draw_mol, round_mol
from .train import prepare_data

import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

sig = nn.Sigmoid()
soft = nn.Softmax(dim=2)
soft0 = nn.Softmax(dim=0)

def start_from(data):

    N = config.max_size

    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    # Get to correct shape
    atom_fea = torch.zeros(N, config.n_onehot+1)
    atom_fea[:data.x.shape[0],:config.n_onehot] = 1*data.x[:,:config.n_onehot]
    atom_fea[data.x.shape[0]:, config.n_onehot] = 1
    
    atom_fea = atom_fea.unsqueeze(0)
    
    #N = atom_fea.shape[1]
    
    adj = torch.zeros(N,N)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5]))

    adj_vec = torch.zeros(N*(N-1)//2)
    for i, t in enumerate(tidx.T):
        adj_vec[i] = adj[t[0],t[1]]
        
    adj_vec = torch.sqrt(adj_vec)
        
    return atom_fea, adj_vec

def smooth_round(x):
    # return (396*torch.pi*x - 225*torch.sin(2*torch.pi*x) + 45*torch.sin(4*torch.pi*x) - 5*torch.sin(6*torch.pi*x))/(300*torch.pi)
    return torch.round(x) + 0.1*(x - torch.round(x))

def max_round(x):

    idx = torch.argmax(x[0,:,:], dim=1)
    
    n_x = 0.1*x
    
    for i,j in enumerate(idx):
        n_x[0,i,j] = 1 - 0.1*(1-x[0,i,j])
    return n_x

    # return x

def weights_to_model_inputs(fea_h, adj_vec, config):

    N = fea_h.shape[1]
    
    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    atom_fea = soft(config.sig_strength*(fea_h-0.5))

    mr_atom_fea = max_round(atom_fea)
    
    atom_fea_ext = torch.cat([mr_atom_fea, torch.matmul(mr_atom_fea[0,:,:config.n_onehot], torch.tensor([[[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]]))],dim=2)
    
    adj = torch.zeros((N,N))
    
    adj[tidx[0],tidx[1]] = adj_vec**2
    
    adj = adj + adj.transpose(0,1)

    adj = smooth_round(adj)
    
    r_features, r_adj = round_mol(atom_fea_ext, adj, config.n_onehot)

    bonds_per_atom = torch.matmul(r_features[:,:config.n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0]))

    constraints = torch.sum((torch.sum(adj, dim=1) - bonds_per_atom)**2)
    
    integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:config.n_onehot+1])**2)

    integer_adj = torch.sum((r_adj - adj)**2)

    # integer_fea = torch.sum(torch.sin(atom_fea_ext[0,:,:config.n_onehot+1]*torch.pi)**6)

    # integer_adj = torch.sum(torch.sin(adj*torch.pi)**6)
    
    adj = adj.unsqueeze(0)

    return atom_fea_ext, adj, constraints, integer_fea, integer_adj

def initialize(qm9, config, output, i):

    N = config.max_size
    
    if config.start_from == "random":
        adj_vec = torch.randn((N*(N-1)//2))
        fea_h = torch.randn((1, N, config.n_onehot+1))
        pickle.dump((adj_vec, fea_h), open(output + "/save_random_%d.pkl"%(i),"wb"))
    elif config.start_from == "saved":
        adj_vec, fea_h = pickle.load(open(output + "/save_random_%d.pkl"%(i),"rb"))
    else:
        qmid = int(config.start_from)
        fea_h, adj_vec = start_from(qm9[qmid])
        print("Model estimate for existing mol:", model(*prepare_data(qm9[qmid], config.max_size, config.one_hot)), config.property_mode_training.max_size)
    
        init_features, init_adj = prepare_data(qm9[qmid], config.max_size, config.one_hot)
    
        mol = MolFromGraph(init_features[0,:,:config.n_onehot],init_adj.squeeze())
    
        img = MolToImage(mol)
    
        img.save(output+"/initial_mol.png")
    
        
    print("INIT ADJ_VEC", adj_vec)
    print("INIT FEA_H", fea_h)
    
    adj_vec.requires_grad = True
    fea_h.requires_grad = True

    return fea_h, adj_vec 
    
def invert(target_value, model, fea_h, adj_vec, config, output):
    """ Backpropagates all the way to the input to produce a molecule with a specific property """

    # Freezes the trained model
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)

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
    cdip = 0
    ccount = 0
    lcount = 0

    if config.show_losses:
        loop = range(config.n_iter)
    else:
        loop = tqdm(range(config.n_iter))
    
    for e in loop:
        atom_fea_ext, adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec, config)
        
        score = model(atom_fea_ext, adj)
        
        loss = abs(score - torch.tensor([[target_value]]))

        n_dip = e - dip
        
        losses_floats = [n_dip/100*float(loss), float(constraints), 0*float(integer_fea), 0*float(integer_adj)]
            
        c = np.array(losses_floats)/np.sum(losses_floats)

        if loss < config.stop_loss:
            dip = e
            c[0] = lcount/10000
            lcount += 1
        else:
            lcount = 0
            
        if constraints < config.stop_chem:
            c[1] = ccount/10000
            cdip = e
            ccount += 1
        else:
            ccount = 0
            
        if constraints < config.stop_chem and loss < config.stop_loss:
            break
        
        total_loss = c[0]*loss + c[1]*constraints + c[2]*integer_fea + c[3]*integer_adj


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
            draw_mol(atom_fea_ext, adj, config.n_onehot, output)

    if config.show_losses:
        plt.ioff()
        
    return fea_h, adj_vec
