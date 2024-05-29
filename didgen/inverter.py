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
    atom_fea = torch.zeros((N,len(config.type_list)), device=device)
    atom_fea[:data.x.shape[0], :] = data.x[:,:len(config.type_list)]
    
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
        return torch.round(x) + 0.001*(x - torch.round(x))

def smooth_max(x):

    idx = torch.argmax(x[:,:], dim=1)
    
    n_x = 0.03*x
    
    for i,j in enumerate(idx):
        n_x[i,j] = 1 - 0.01*(1-x[i,j])
    return n_x

    # return x
    
def bell(x,p):
    sig = 0.8
    m=0.8
    b=0.2
    x0 = sig*torch.sqrt(-torch.log(torch.tensor(0.05)))
    return (torch.exp(-((x - p)/sig)**2)*m + b)*((x-p < x0) & (x-p > -x0)) + (x-p > x0)*(-m*2*(x0)/sig**2*torch.exp(-((x0)/sig)**2)*(x - x0 - p) + m*torch.exp(-((x0)/sig)**2) + b) + (x-p < -x0)*(m*2*(x0)/sig**2*torch.exp(-((x0)/sig)**2)*(x + x0 - p) + m*torch.exp(-((x0)/sig)**2) + b)
    #return torch.exp(-((x - p)/sig)**2)
    #return (-abs(x - p) + 1.1)*(abs(x - p) < 1) + (-0.1*(x-p-1) + 0.1)*((x > p + 1) & (x < p + 2)) + (0.1*(x-p+1) + 0.1)*((x < p - 1) & (x > p - 2))

def smooth_onehot(x, p):
    return bell(x,p)*((x < 4) if p==4 else 1) + (1*(x >= 4) if p==4 else 0)
    # return torch.cos(torch.pi/2*(x-p))**2 * torch.heaviside(x-p+1,torch.tensor([0.0])) * (torch.heaviside(-x+p+1,torch.tensor([0.0])) if p < 4 else torch.heaviside(-x+p,torch.tensor([0.0]))) + (0 if p < 4 else torch.heaviside(x-p,torch.tensor([0.0])))
    #return torch.cos(torch.pi/2*(x-p))**2*((x >= p-1) & (x < p+1))*(x < 4) + (1*(x >= 4) if p==4 else 0)
    
def weights_to_model_inputs(fea, adj_vec, config, adj_only=False):
    
    device = adj_vec.device
    
    N = len(fea)
    
    tidx = torch.tril_indices(row=N, col=N, offset=-1)

    tid_mask = torch.tril_indices(row=N, col=N, offset=-7) #TMP 
    
    adj = torch.zeros((N,N), device=device)
    
    adj[tidx[0],tidx[1]] = adj_vec**2 #* config.adj_mask

    # adj[tid_mask[0],tid_mask[1]] = 0 #TMP
    
    adj = adj + adj.transpose(0,1)
    
    adj = smooth_round(adj, method=config.rounding)

    if adj_only:
        return adj.unsqueeze(0)
    
    adj_sum = torch.sum(adj, dim=1).unsqueeze(1)
    
    fea_h = torch.zeros(fea.shape, device=device)
 
    # for i in set(config.bonding):
    #     fea_h[:, config.bonding == i] = smooth_max(fea[:, config.bonding == i]**2/torch.sum(fea[:, config.bonding == i]**2,dim=1,keepdim=True))
    
    for b in config.bonding_mask:
        if sum(b) == 1:
            fea_h[:, b] = 1
        elif sum(b) == 2:
            # fea_h[:, b] = torch.cat([sig((2*fea[:,b][:,0] - 1.0)).unsqueeze(1), 1-sig((2*fea[:,b][:,0] - 1.0)).unsqueeze(1)], dim=1)#
            fea_h[:, b] = smooth_round(fea[:, b]**2/torch.sum(fea[:, b]**2 + 1e-12,dim=1,keepdim=True))
        else:
            fea_h[:, b] = smooth_max(fea[:, b]**2/torch.sum(fea[:, b]**2 + 1e-12,dim=1,keepdim=True))
            
    # print(adj_sum)

    # print(fea_h)

    # input()
            
    # isH = sig((2*fea[:,0] - 1.0)).unsqueeze(1)

    # atom_fea_adj = torch.cat([smooth_onehot(adj_sum, 1.0)*isH, smooth_onehot(adj_sum, 4.0), smooth_onehot(adj_sum, 3.0), smooth_onehot(adj_sum, 2.0), smooth_onehot(adj_sum, 1.0)*(1-isH), smooth_onehot(adj_sum, 0.0)], dim=1)

    #print("INSIDE")
    #print(adj_sum)

    # for i,t in enumerate(config.type_list):
    #     print(i, t, config.bonding[i], smooth_onehot(adj_sum, config.bonding[i])[:10])
    
    atom_fea_adj = torch.cat([smooth_onehot(adj_sum, config.bonding[i]) for i,t in enumerate(config.type_list)], dim=1) 

    #print("From adj")
    #print(atom_fea_adj[:10,:])

    #print("Fea")
    #print(fea_h[:10,:])    

    atom_fea_adj = atom_fea_adj * fea_h

    atom_fea_adj = torch.cat([atom_fea_adj, smooth_onehot(adj_sum, 0.0)], dim=1)

    #print("before max", atom_fea_adj)
    
    atom_fea_adj = smooth_max(atom_fea_adj)
    
    atom_fea_ext = add_extra_features(atom_fea_adj, config._extra_fea_matrix).unsqueeze(0)

    return atom_fea_ext, adj.unsqueeze(0)

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

    os.makedirs(output + "/save_random", exist_ok=True)
    
    if config.start_from == "random":
        
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
        
        def keep_in(data):
            return len(data.x) <= config.max_size
        
        assert len(config.datasets) == 1, "Need only one dataset to start from" 

        dset = config.datasets[0]
        
        if dset == "qm9":

            from torch_geometric.datasets import QM9
            
            dataset = QM9(output + "/" + dset, pre_filter=keep_in)
        
            if ["H", "C", "N", "O", "F"] != config.type_list:
                raise RuntimeError("type_list is incompatible with dataset", list(dataset_dict["qm9"].types)) 
        else:

            from .custom_dataset import QM9like
            
            dataset = QM9like(output + "/" + dset, pre_filter=keep_in)
        
            if list(dataset.types) != config.type_list:
                raise RuntimeError("type_list is incompatible with dataset", list(dataset.types))


        if config.start_from == "!":
            starting_tries = 1000
            
            size = config.starting_size + 1
            tries = 0
            while size > config.starting_size and tries < starting_tries:
                qmid = int(torch.randint(len(dataset),(1,)))
                size = len(dataset[qmid].x)

            if tries == starting_tries:
                raise RuntimeError("Could not find a molecule of the right size in dataset")
        else:
            qmid = int(config.start_from)
            if len(dataset[qmid].x) > config.starting_size:
                raise RuntimeError("Molecule larger than starting_size")
        
        fea_h, adj_vec = start_from(dataset[qmid].to(device), config)

        mask = (adj_vec == 0)
        mask_mask = (torch.rand(mask.shape) > 0.15)
        config.adj_mask = ~(mask * mask_mask)
        
        # nudge

        N = config.max_size

        fea_h = fea_h + torch.randn(fea_h.shape, device=device)*config.adj_eps
        
        adj_vec = adj_vec + torch.randn((N*(N-1)//2), device=device)*config.adj_eps

        pickle.dump((adj_vec, fea_h), open(output + "/save_random/%d.pkl"%(i),"wb")) # Save the random full matrices
        
        init_features, init_adj = prepare_data(dataset[qmid].to(device), config.max_size, config._extra_fea_matrix)
        
        mol = MolFromGraph(init_features[0,:,:len(config.type_list)],init_adj.squeeze(), config.type_list)

        pickle.dump(mol, open(output + "/save_random/%d.mol.pkl"%(i),"wb"))
        
        img = MolToImage(mol)
    
        img.save(output+"/initial_mol.png")
    
        
    #print("INIT ADJ_VEC", adj_vec)
    #print("INIT FEA_H", fea_h)
    
    adj_vec.requires_grad = True
    fea_h.requires_grad = True

    return fea_h, adj_vec, config

def invert(model, fea_h, adj_vec, config, output):
    """ Backpropagates all the way to the input to produce a molecule with a specific property """

    # Freezes the trained model
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([{"params":adj_vec, "lr": config.inv_r}, {"params":fea_h, "lr": config.inv_r}])
    #optimizer = optim.SGD([adj_vec, fea_h], lr=config.inv_r)
    #optimizer = optim.RMSprop([adj_vec, fea_h], lr=config.inv_r)

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
        
        proportions = (torch.sum(atom_fea_ext[0,:,:len(config.type_list)], dim=0)/torch.sum(atom_fea_ext[0,:,:len(config.type_list)]) - torch.tensor(config.proportions, device=score.device))**2
        
        proportions = torch.sum(proportions)
        
        # proportions = (torch.sum(atom_fea_ext[0,:,:len(config.type_list)], dim=0) - torch.tensor(config.proportions, device=score.device))**2
        
        # objective = torch.zeros((config.max_size, len(config.type_list)+1))

        # p=0
        # for i, num in enumerate(config.proportions):
        #     for j in range(num):
        #         objective[p,i] = 1
        #         p+=1
        
        # objective[p:,len(config.type_list)] = 1

        #print(torch.sum(adj, dim=1))
        #print(torch.matmul(objective[:,:len(config.type_list)], config.bonding*1.0))
        
        # proportions = (torch.sum(adj, dim=1) - torch.matmul(objective[:,:len(config.type_list)], config.bonding*1.0))**2

        adj_sum = torch.sum(adj, dim=1).squeeze()
        
        # N_proportions = smooth_onehot(adj_sum, 3) #+ smooth_onehot(adj_sum, 2)

        # N_proportions = N_proportions[N_proportions > 0.5]

        # print("N", N_proportions)

        # adj_c = torch.zeros(adj.shape)
        
        # adj_c[:,:,((adj_sum > 0) & (adj_sum < 0.5))] = adj[:,:,((adj_sum > 0) & (adj_sum < 0.5))]

        # #adj_c[:,:,((adj_sum > 1.5) & (adj_sum < 3.5))] = adj[:,:,((adj_sum > 1.5) & (adj_sum < 3.5))]

        # # adj_c_sum = torch.max(adj_c, dim=2).values.squeeze()

        # adj_c_sum = torch.sum(adj_c, dim=2).squeeze()
        
        # proportions_N = 1 - adj_c_sum[(adj_sum > 2.5) & (adj_sum < 3.5)] + torch.round(adj_c_sum[(adj_sum > 2.5) & (adj_sum < 3.5)])

        # proportions_O = 1 - adj_c_sum[(adj_sum > 1.5) & (adj_sum < 2.5)] + torch.round(adj_c_sum[(adj_sum > 1.5) & (adj_sum < 2.5)])
                
        #O_proportions = smooth_onehot(adj_sum, 2) #+ smooth_onehot(adj_sum, 2)

        #O_proportions = O_proportions[O_proportions > 0.5]

        #print(proportions_O)

        #print(proportions_N)

        # proportions = (atom_fea_ext[0,:,:len(config.type_list)+1] - objective)**2

        #print(atom_fea_ext[0,:,:len(config.type_list)+1])

        #print(objective)
        
        #input()
        
        #proportions = proportions if proportions > 5 else 0

        #proportions = torch.sum(proportions[proportions > 0.04])

        #proportions = torch.sum(proportions_O) + torch.sum(proportions_N)

        #input()

        #print(atom_fea_ext[0,:,1], torch.sum(atom_fea_ext[0,:,1], dim=0))

        #input()
        
        #proportions = config.max_size - torch.sum(atom_fea_ext[0,:,0], dim=0)
        
        # if init:
        #     crit = 0.02
        #     loss = proportions

        # if e>400:
        #     print(atom_fea_ext[0,:,:len(config.type_list)+1])
        #     print(torch.sum(adj, dim=1))
        #     input()
        
    
        constraints = torch.sum(torch.sum(adj, dim=1)[torch.sum(adj, dim=1) > 4.5])
        
        r_features, r_adj = round_mol(atom_fea_ext, adj, len(config.type_list))
        
        integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:len(config.type_list)+1])**2)
        
        integer_adj = torch.sum((r_adj - adj.squeeze())**2)
        
        r_bonds_per_atom = torch.matmul(r_features, torch.cat([config.bonding, torch.zeros(1, device=r_features.device)]))            

        n_blank = int(torch.sum(r_features[:,len(config.type_list)]))

        # if n_blank > config.max_size - config.min_size:
        #     print("Too small!", n_blank)
        
        # proportions = proportions*(proportions > config.stop_prop*(config.max_size - n_blank))
        
        if config.true_loss:
            
            true_score = model(add_extra_features(r_features, config._extra_fea_matrix).unsqueeze(0), r_adj.unsqueeze(0)).detach()
            
            true_loss = float(torch.sum(abs(true_score - torch.tensor([config.target], device=true_score.device))))

        else:
            
            true_loss = loss

        # if loss < 1e-6:
        #     print("rounded")
        #     print(r_features)
        #     print("features")
        #     print(atom_fea_ext[0,:,:len(config.type_list)+1])
        #     print(r_bonds_per_atom)
        #     print(torch.sum(r_adj, dim=1))
        #     print(torch.sum(adj.squeeze(), dim=1))
        #     input()


        if not (torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12) and constraints < 1e-12:
            print("PROBLEM bonding is bad but no constraint", torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))), constraints)
            #print(adj_vec)
            #print(adj)
            print(r_bonds_per_atom)
            print(torch.sum(r_adj, dim=1))
            #print(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1)))
            #input()

        if torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12 and (true_loss < config.stop_loss or loss < 1e-6):
            print("LOSS REACHED, prop:", proportions)
            
        if torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12 and (true_loss < config.stop_loss or loss < 1e-6) and proportions < config.stop_prop: #crit:

            # Number of components in graph
            L = torch.diag(torch.sum((r_adj != 0),axis=0)) - (r_adj != 0)*1
            try:
                n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - n_blank
            except:
                print(L.float(), L.max(), L.min())
                draw_mol(atom_fea_ext, adj, config.type_list, output, text=" ".join(["%f"]*(len(score[0])))%tuple(score[0].detach().tolist()), color=(int(255*(float(loss/sum([abs(e) for e in config.target])))),int(255*(1-float(loss/sum([abs(e) for e in config.target])))),0))
                n_comp = 2
                
            if n_comp == 1 and config.max_size - n_blank >= config.min_size:
                # if init:
                #     init = False
                #     crit = config.stop_loss
                #     draw_mol(atom_fea_ext, adj, config.type_list, output, text="%f"%(score), color=(int(255*(float(loss/score))),int(255*(1-float(loss/score))),0))
                #     print((torch.sum(atom_fea_ext[0,:,:len(config.type_list)], dim=0)/config.max_size - torch.tensor([0.4110, 0.3026, 0.0602, 0.0735, 0.0015]))**2)
                #     input()
                #     optimizer = optim.Adam([adj_vec, fea_h], lr=config.inv_r)
                #     continue
                print("FINAL PROP:")
                print(proportions)
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

                        #N = config.max_size
                        
                        #adj_vec = adj_vec + torch.randn((N*(N-1)//2), device=score.device)*config.adj_eps
                                        
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

        #total_loss = (config.l_loss*loss**2 + config.l_const*constraints**2)/(config.l_loss*loss + config.l_const*constraints)

        #total_loss = config.l_loss*loss + config.l_const*constraints + config.l_prop*proportions

        #total_loss = (config.l_loss*loss.detach()*loss + config.l_const*constraints.detach()*constraints + config.l_prop*proportions.detach()*proportions)/(config.l_loss*loss.detach() + config.l_const*constraints.detach() + config.l_prop*proportions.detach())

        total_losses.append(float(total_loss))
        losses.append(float(loss))
        constraint_losses.append(float(constraints))
        integer_fea_losses.append(float(integer_fea))
        #integer_adj_losses.append(float(integer_adj))
        integer_adj_losses.append(float(proportions))
        
        # backward

        #grads_l = torch.autograd.grad(total_loss,[adj_vec, fea_h], retain_graph=True)        
        #grads_p = torch.autograd.grad(config.l_prop*proportions,[adj_vec], retain_graph=True)
        #grads_c = torch.autograd.grad(config.l_const*constraints,[adj_vec], retain_graph=True)

        adj_vec_before = 1*adj_vec
        
        optimizer.zero_grad()
        total_loss.backward()

        # N = config.max_size

        # tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
        # adj_indices = -torch.ones((N,N), device=adj_vec.device, dtype=torch.long)
    
        # adj_indices[tidx[0],tidx[1]] = torch.arange(adj_vec.shape[0])

        # adj_indices[tidx[1],tidx[0]] = torch.arange(adj_vec.shape[0])
        
        # id_dont_move = adj_indices[:,adj_sum > 3.5]

        # id_dont_move = torch.flatten(id_dont_move[id_dont_move != -1])

        # adj_vec.grad[id_dont_move[adj_vec.grad[id_dont_move] < 0]] = 0


        # id_dont_move = adj_indices[:,(adj_sum > 1.5) & (adj_sum < 3.5)]

        # id_dont_move = torch.flatten(id_dont_move[id_dont_move != -1])

        # adj_vec.grad[id_dont_move[adj_vec.grad[id_dont_move] < 0]] = 0

        # adj_vec.grad[id_dont_move[adj_vec.grad[id_dont_move] > 0]] = adj_vec.grad[id_dont_move[adj_vec.grad[id_dont_move] > 0]] * 10

        # if proportions > 0:
        
        #     grad_p_adj = torch.zeros(grads_p[0].shape)

        #     p_condition = (torch.sign(grads_p[0]) == torch.sign(grads_l[0]))

        #     grad_p_adj[p_condition] = grads_p[0][p_condition]

        #     #print("PROP grad", grad_p_adj.max(), grad_p_adj.min())
        
        #     grad = grads_l[0] + grad_p_adj

        # else:

        #     grad = grads_l[0]
        

        #print("CONST grad", len(grad[(torch.sign(grad) != torch.sign(grads_c[0])) & (torch.sign(grads_c[0]) != 0)])/(config.max_size-1))

        #print(constraints)

        #c_condition = ((torch.sign(grad) != torch.sign(grads_c[0])) & (torch.sign(grads_c[0]) != 0))
        
        #grad[c_condition] = 0
        
        # adj_vec.grad = grad #+ grads_c[0]

        # fea_h.grad = grads_l[1]
        
        # print(grad.max(), grad.min())

        # print(grads_l[1].max(), grads_l[1].min())

        # print(adj)
        
        # input()
        
        # gradient descent or adam step
        optimizer.step()

        adj_tmp = weights_to_model_inputs(fea_h, adj_vec, config, adj_only=True)

        old_adj_sum = adj_sum
        
        adj_sum = torch.sum(adj_tmp, dim=1).squeeze()

        if sum(adj_sum > 4.5) > 0:

            # print("e in", e)
            
            N = config.max_size
            
            tidx = torch.tril_indices(row=N, col=N, offset=-1)
            
            adj_indices = -torch.ones((N,N), device=adj_vec.device, dtype=torch.long)
            
            adj_indices[tidx[0],tidx[1]] = torch.arange(adj_vec.shape[0])
            
            adj_indices[tidx[1],tidx[0]] = torch.arange(adj_vec.shape[0])

            adj_diff = torch.round(adj_tmp - adj)
            
            adj_overflow = torch.round(adj_sum - 4)
            
            adj_culprits = torch.zeros(adj.shape)

            # Less than 4 bonds ------------

            adj_culprits[((adj_overflow.unsqueeze(1).expand(N,N).unsqueeze(0) > 0) & (adj_diff > 0))] = adj_diff[(adj_overflow.unsqueeze(1).expand(N,N).unsqueeze(0) > 0) & (adj_diff > 0)]
            
            sorted_culprits, indices_x = torch.sort(adj_culprits, dim=2)
            
            indices_y = torch.arange(N).unsqueeze(1).expand(N,N).unsqueeze(0)
            
            cumsum = torch.cumsum(sorted_culprits, dim=2)
            
            idx = indices_x[(cumsum < adj_overflow.unsqueeze(1).expand(N,N).unsqueeze(0) + sorted_culprits) & (cumsum > 0)]

            idy = indices_y[(cumsum < adj_overflow.unsqueeze(1).expand(N,N).unsqueeze(0) + sorted_culprits) & (cumsum > 0)]
            
            id_dont_move = adj_indices[idy,idx]
            
            id_dont_move = torch.flatten(id_dont_move[id_dont_move != -1])

            with torch.no_grad():
            
                adj_vec[id_dont_move] = adj_vec_before[id_dont_move] #+ 0.001*(adj_vec[id_dont_move] - adj_vec_before[id_dont_move])

                
        adj_tmp = weights_to_model_inputs(fea_h, adj_vec, config, adj_only=True)
        
        adj_sum = torch.sum(adj_tmp, dim=1).squeeze()
                
        # More than min_size atoms --------------------

        if False: #sum(adj_sum < 0.5) > config.max_size - config.min_size:

            #print("Going to be too small:", sum(adj_sum < 0.5))
            
            N = config.max_size
            
            tidx = torch.tril_indices(row=N, col=N, offset=-1)
            
            adj_indices = -torch.ones((N,N), device=adj_vec.device, dtype=torch.long)
            
            adj_indices[tidx[0],tidx[1]] = torch.arange(adj_vec.shape[0])
            
            adj_indices[tidx[1],tidx[0]] = torch.arange(adj_vec.shape[0])

            adj_diff = torch.round(adj_tmp - adj)
            
            adj_culprits = torch.zeros(adj.shape)
           
            adj_culprits = torch.zeros(adj.shape)
           
            adj_culprits[(adj_sum.unsqueeze(1).expand(N,N).unsqueeze(0) < 0.5) & (adj_diff < 0)] = -adj_diff[(adj_sum.unsqueeze(1).expand(N,N).unsqueeze(0) < 0.5) & (adj_diff < 0)]
           
            sorted_culprits, indices = torch.sort(torch.flatten(adj_culprits))
           
            cumsum = torch.cumsum(sorted_culprits, dim=0)
            
            idx = indices[(cumsum <  sum(adj_sum < 0.5) - config.max_size + config.min_size + sorted_culprits) & (cumsum > 0)]
            
            id_dont_move = adj_indices[idx//adj.shape[1], idx%adj.shape[1]]
            
            id_dont_move = torch.flatten(id_dont_move[id_dont_move != -1])
            
            with torch.no_grad():
           
                adj_vec[id_dont_move] = adj_vec_before[id_dont_move] #+ 0.001*(adj_vec[id_dont_move] - adj_vec_before[id_dont_move])

            _, adj_tmp = weights_to_model_inputs(fea_h, adj_vec, config)
        
            adj_sum = torch.sum(adj_tmp, dim=1).squeeze()

            # if sum(adj_sum < 0.5) > config.max_size - config.min_size:

            #     #print(cumsum)

            #     #print(id_dont_move)

            #     print("Still too small:", sum(adj_sum < 0.5))

            #     #input()
                
        if e%10 == 0 and config.show_losses:
            #print(c[0], c[1], n_dip, float(loss), torch.max(abs(fea_h)), torch.max(abs(adj_vec)))
            print(float(total_loss), float(loss), "(", true_loss, ")", float(proportions), n_blank, score.detach().tolist(), torch.sum(abs(r_bonds_per_atom - torch.sum(r_adj, dim=1))) < 1e-12, (true_loss < config.stop_loss or loss < 1e-6), proportions < config.stop_prop)
            
        if e%30 == 0 and config.show_losses:
            plt.plot(total_losses,'k')
            plt.plot(losses,'b')
            plt.plot(constraint_losses,".-", color="r")
            plt.plot(integer_fea_losses, color="orange")
            plt.plot(integer_adj_losses, color="purple")
            plt.plot(g1s, color="pink")
            plt.plot(g2s, color="green")
            plt.pause(0.1)
            draw_mol(atom_fea_ext, adj, config.type_list, output, text=" ".join(["%f"]*(len(score[0])))%tuple(score[0].detach().tolist()), color=(int(255*(float(loss/sum([abs(e) for e in config.target])))),int(255*(1-float(loss/sum([abs(e) for e in config.target])))),0))

        #input()
            
    if config.show_losses:
        plt.ioff()
        plt.show()

    # print(fea_h, adj_vec)
        
    return fea_h, adj_vec, e
