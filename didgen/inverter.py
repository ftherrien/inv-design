from .utils import draw_mol, round_mol, MolFromGraph
from .train import prepare_data, add_extra_features

import torch
from torch import optim, nn
from tqdm import tqdm
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
    atom_fea = torch.zeros(N, device=device)
    atom_fea[:data.x.shape[0]][data.x[:,len(config.type_list)] == 1] = 1
    
    adj = torch.zeros(N, N, device=device)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))

    adj_vec = torch.zeros(N*(N-1)//2, device=device)
    for i, t in enumerate(tidx.T):
        adj_vec[i] = adj[t[0],t[1]]
        
    adj_vec = torch.sqrt(adj_vec)
        
    return atom_fea, adj_vec

def smooth_round(x, method="step"):
    if method=="sin":
        return (396*torch.pi*x - 225*torch.sin(2*torch.pi*x) + 45*torch.sin(4*torch.pi*x) - 5*torch.sin(6*torch.pi*x))/(396*torch.pi)
    elif method=="step":
        return torch.round(x) + 0.05*(x - torch.round(x))

def smooth_max(x):

    idx = torch.argmax(x[:,:], dim=1)
    
    n_x = 0.3*x
    
    for i,j in enumerate(idx):
        n_x[i,j] = 1 - 0.3*(1-x[i,j])
    return n_x

    # return x
    
    
def smooth_onehot(x, p):
    return torch.exp(-(x - p)**2/0.5)*((x < 4) if p==4 else 1) + (1*(x >= 4) if p==4 else 0)
    # return torch.cos(torch.pi/2*(x-p))**2*((x >= p-1) & (x < p+1))*(x < 4) + (1*(x >= 4) if p==4 else 0)
    
def weights_to_model_inputs(fea, adj_vec, config):
    
    device = adj_vec.device
    
    N = len(fea)
    
    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    adj = torch.zeros((N,N), device=device)
    
    adj[tidx[0],tidx[1]] = adj_vec**2
    
    adj = adj + adj.transpose(0,1)
    
    adj = smooth_round(adj, method=config.rounding)
    
    adj_sum = torch.sum(adj, dim=1).unsqueeze(1)
    
    fea_h = torch.ones(fea.shape, device=device)
 
    # for i in set(config.bonding):
    #     fea_h[:, config.bonding == i] = smooth_max(fea[:, config.bonding == i]**2/torch.sum(fea[:, config.bonding == i]**2,dim=1,keepdim=True))
    
    for b in config.bonding_mask:
        if sum(b) == 2:
            fea_h[:, b] = torch.cat([sig((2*fea[:,b][:,0] - 1.0)).unsqueeze(1), 1-sig((2*fea[:,b][:,0] - 1.0)).unsqueeze(1)], dim=1)#smooth_max(fea[:, b]**2/torch.sum(fea[:, b]**2,dim=1,keepdim=True))
        else:
            fea_h[:, b] = smooth_max(fea[:, b]**2/torch.sum(fea[:, b]**2,dim=1,keepdim=True))

    # print(adj_sum)
            
    # print(fea_h)

    # input()
            
    # isH = sig((2*fea[:,0] - 1.0)).unsqueeze(1)

    # atom_fea_adj = torch.cat([smooth_onehot(adj_sum, 1.0)*isH, smooth_onehot(adj_sum, 4.0), smooth_onehot(adj_sum, 3.0), smooth_onehot(adj_sum, 2.0), smooth_onehot(adj_sum, 1.0)*(1-isH), smooth_onehot(adj_sum, 0.0)], dim=1)
    
    atom_fea_adj = torch.cat([smooth_onehot(adj_sum, config.bonding[i]) for i,t in enumerate(config.type_list)], dim=1) * fea_h

    atom_fea_adj = torch.cat([atom_fea_adj, smooth_onehot(adj_sum, 0.0)], dim=1)
    
    atom_fea_ext = add_extra_features(atom_fea_adj, config._extra_fea_matrix).unsqueeze(0)
        
    adj = adj.unsqueeze(0)

    return atom_fea_ext, adj 

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


def initialize(config, output, i, device="cpu"):

    N = config.max_size

    if type(config.starting_size) == list:
        max_N = int(torch.randint(*config.starting_size, (1,)))
    else:
        max_N = config.starting_size
        
    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    if config.start_from == "random":
        
        os.makedirs(output + "/save_random", exist_ok=True)
        
        adj_vec = torch.randn((N*(N-1)//2), device=device)*config.bond_multiplier
        
        fea_h = torch.zeros((N,len(config.type_list)),device=device)
        
        for b in config.bonding_mask:
            fea_h[:,b] = torch.rand((N,sum(b)),device=device)

        pickle.dump((adj_vec, fea_h), open(output + "/save_random/%d.pkl"%(i),"wb")) # Save the random full matrices 
        
        # Shorten the input to fit max_N 

        adj_vec[(tidx[0] > max_N - 1) | (tidx[1] > max_N - 1)] = 0
        
        
    elif config.start_from == "saved":
        
        adj_vec, fea_h = pickle.load(open(output + "/save_random/%d.pkl"%(i),"rb"))
        
        # Shorten the input to fit max_N 

        adj_vec[(tidx[0] > max_N - 1) | (tidx[1] > max_N - 1)] = 0
        
        
    else:

        from torch_geometric.datasets import QM9

        def keep_in(data):
            return len(data.x) <= config.max_size
        
        qm9 = QM9(output + "/dataset", pre_filter=keep_in)

        qmid = int(config.start_from)
        
        fea_h, adj_vec = start_from(qm9[qmid].to(device), config)

        # nudge

        N = config.max_size
        
        adj_vec = adj_vec + torch.randn((N*(N-1)//2), device=device)*config.adj_eps
        
        init_features, init_adj = prepare_data(qm9[qmid].to(device), config.max_size, len(config.type_list))
    
        mol = MolFromGraph(init_features[0,:,:len(config.type_list)],init_adj.squeeze(), len(config.type_list))
    
        img = MolToImage(mol)
    
        img.save(output+"/initial_mol.png")
    
        
    #print("INIT ADJ_VEC", adj_vec)
    #print("INIT FEA_H", fea_h)
    
    adj_vec.requires_grad = True
    fea_h.requires_grad = True

    return fea_h, adj_vec

def invert(model, fea_h, adj_vec, config, output):
    """ Backpropagates all the way to the input to produce a molecule with a specific property """
    
    # Freezes the trained model
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([{"params":adj_vec, "lr": config.inv_r}, {"params":fea_h, "lr": config.inv_r}])
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

    #init = True
    for e in loop:
        
        atom_fea_ext, adj = weights_to_model_inputs(fea_h, adj_vec, config)
        
        score = model(atom_fea_ext, adj)
        
        loss = torch.sum(abs(score - torch.tensor([config.target], device=score.device)))

        proportions = (torch.sum(atom_fea_ext[0,:,:len(config.type_list)], dim=0)/config.max_size - torch.tensor(config.proportions, device=score.device))**2

        #proportions = proportions if proportions > 5 else 0

        #proportions = torch.sum(proportions[proportions > 0.04])

        proportions = torch.sum(proportions)

        #proportions = config.max_size - torch.sum(atom_fea_ext[0,:,0], dim=0)
        
        # if init:
        #     crit = 0.02
        #     loss = proportions
        
        constraints = torch.sum(torch.sum(adj, dim=1)[torch.sum(adj, dim=1) > 4.3])

        r_features, r_adj = round_mol(atom_fea_ext, adj, len(config.type_list))
        
        integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:len(config.type_list)+1])**2)
        
        integer_adj = torch.sum((r_adj - adj.squeeze())**2)
        
        r_bonds_per_atom = torch.matmul(r_features, torch.cat([config.bonding, torch.zeros(1, device=r_features.device)]))            

        true_score = model(add_extra_features(r_features, config._extra_fea_matrix).unsqueeze(0), r_adj.unsqueeze(0)).detach()

        true_loss = float(torch.sum(abs(true_score - torch.tensor([config.target], device=true_score.device))))
        
        if torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12 and (true_loss < config.stop_loss or loss < 1e-6) and proportions < config.stop_prop: #crit:

            # Number of components in graph
            L = torch.diag(torch.sum((r_adj != 0),axis=0)) - (r_adj != 0)*1
            n_blank = int(torch.sum(r_features[:,len(config.type_list)]))
            n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - n_blank

            if n_comp == 1 and config.max_size - n_blank >= config.min_size:
                # if init:
                #     init = False
                #     crit = config.stop_loss
                #     draw_mol(atom_fea_ext, adj, config.type_list, output, text="%f"%(score), color=(int(255*(float(loss/score))),int(255*(1-float(loss/score))),0))
                #     print((torch.sum(atom_fea_ext[0,:,:len(config.type_list)], dim=0)/config.max_size - torch.tensor([0.4110, 0.3026, 0.0602, 0.0735, 0.0015]))**2)
                #     input()
                #     optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
                #     continue
                break
            else:

                print("TOO MANY COMPS!")
                
                #draw_mol(atom_fea_ext, adj, config.type_list, output)
                
                comps = get_components(r_adj)
                idx = comps[torch.argmax(torch.tensor([len(c) for c in comps]))]

                if len(idx) >= config.min_size:
                
                    tidx = torch.tril_indices(row=r_adj.shape[0], col=r_adj.shape[0], offset=-1)
                    
                    with torch.no_grad():
                        
                        adj_vec[torch.all(tidx[0].unsqueeze(1).expand(-1,len(idx)) != torch.tensor([idx]).expand(len(tidx[0]),-1), dim=1) & torch.any(tidx[1].unsqueeze(1).expand(-1,len(idx)) != torch.tensor([idx]).expand(len(tidx[1]),-1), dim=1)] = 0
                        
                                        
                    adj_vec, fea_h = adj_vec.detach(), fea_h.detach()
                    
                    adj_vec.requires_grad = True
                    fea_h.requires_grad = True
                    
                    optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
                    
                    continue

                else:

                    print("SMALL COMP!")

                    with torch.no_grad():

                        N = config.max_size

                        adj_vec = adj_vec + torch.randn((N*(N-1)//2), device=score.device)*config.adj_eps
                        
                    adj_vec, fea_h = adj_vec.detach(), fea_h.detach()
                    
                    adj_vec.requires_grad = True
                    fea_h.requires_grad = True
                    
                    optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
                    
                    continue
        
        total_loss = (config.l_loss*loss**2 + config.l_const*constraints**2 + config.l_prop*proportions**2)/(config.l_loss*loss + config.l_const*constraints + config.l_prop*proportions)

        #total_loss = (config.l_loss*loss.detach()*loss + config.l_const*constraints.detach()*constraints + config.l_prop*proportions.detach()*proportions)/(config.l_loss*loss.detach() + config.l_const*constraints.detach() + config.l_prop*proportions.detach())

        total_losses.append(float(total_loss))
        losses.append(float(loss))
        constraint_losses.append(float(constraints))
        integer_fea_losses.append(float(integer_fea))
        #integer_adj_losses.append(float(integer_adj))
        integer_adj_losses.append(float(proportions))
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
        
        if e%10 == 0 and config.show_losses:
            #print(c[0], c[1], n_dip, float(loss), torch.max(abs(fea_h)), torch.max(abs(adj_vec)))
            print(float(total_loss), float(loss), float(proportions), score.detach().tolist(), torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12, (true_loss < config.stop_loss or loss < 1e-6), proportions < config.stop_prop)
            
        if e%1000 == 0 and config.show_losses:
            plt.plot(total_losses,'k')
            plt.plot(losses,'b')
            plt.plot(constraint_losses,".-", color="r")
            plt.plot(integer_fea_losses, color="orange")
            plt.plot(integer_adj_losses, color="purple")
            plt.plot(g1s, color="pink")
            plt.plot(g2s, color="green")
            plt.pause(0.1)
            draw_mol(atom_fea_ext, adj, config.type_list, output, text=" ".join(["%f"]*(len(score[0])))%tuple(score[0].detach().tolist()), color=(int(255*(float(loss/sum([abs(e) for e in config.target])))),int(255*(1-float(loss/sum([abs(e) for e in config.target])))),0))
            
    if config.show_losses:
        plt.ioff()

    # print(fea_h, adj_vec)
        
    return fea_h, adj_vec, e
