import torch
from torch import optim, nn  
from tqdm import tqdm  # For nice progress bar!
import numpy as np
import pickle

# From Alex
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.Draw import MolToImage
import rdkit.Chem as Chem

from model import CrystalGraphConvNet as CGCNN
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import mdmm
from multi_task.min_norm_solvers import MinNormSolver, gradient_normalizers

torch.set_printoptions(sci_mode=False)

## HYPERPARAMETERS =================

n_data = 690
num_epochs = 500
batch_size = 230
orig_atom_fea_len = 5
nbr_fea_len = 4
learning_rate = 0.01
max_size = 10
use_pretrained = True
random_split = True
sig_strength = 1 #7
target_value = 2.7
from_random = True
from_saved_random = True
qmid = 14 #(7.29)
n_iter = 60000
inv_r = 0.001
n_onehot = 5
l=1
l_fea=0
l_adj=0.3
noise_factor = 0.2
stop_chem = 0.1
stop_loss = 0.3

# ==================================

sig = nn.Sigmoid()
soft = nn.Softmax(dim=2)
soft0 = nn.Softmax(dim=0)

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

def weights_to_model_inputs(fea_h, adj_vec):

    N = fea_h.shape[1]
    
    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    atom_fea = soft(sig_strength*(fea_h-0.5))

    mr_atom_fea = max_round(atom_fea)
    
    atom_fea_ext = torch.cat([mr_atom_fea, torch.matmul(mr_atom_fea[0,:,:n_onehot], torch.tensor([[[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]]))],dim=2)
    
    adj = torch.zeros((N,N))
    
    adj[tidx[0],tidx[1]] = adj_vec**2
    
    adj = adj + adj.transpose(0,1)

    adj = smooth_round(adj)
    
    r_features, r_adj = round_mol(atom_fea_ext, adj)

    bonds_per_atom = torch.matmul(r_features[:,:n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0]))

    constraints = torch.sum((torch.sum(adj, dim=1) - bonds_per_atom)**2)
    
    integer_fea = torch.sum((r_features - atom_fea_ext[0,:,:n_onehot+1])**2)

    integer_adj = torch.sum((r_adj - adj)**2)

    # integer_fea = torch.sum(torch.sin(atom_fea_ext[0,:,:n_onehot+1]*torch.pi)**6)

    # integer_adj = torch.sum(torch.sin(adj*torch.pi)**6)
    
    adj = adj.unsqueeze(0)

    return atom_fea_ext, adj, constraints, integer_fea, integer_adj

def round_mol(atom_fea_ext, adj, smooth=False, half=False):

    N = atom_fea_ext.shape[1]

    idx = torch.argmax(atom_fea_ext[0,:,:n_onehot+1], dim=1)
    
    features = torch.zeros((N,n_onehot+1))
    
    for i,j in enumerate(idx):
        features[i,j] = 1
    
    if smooth:
        adj = smooth_round(adj).squeeze()
    else:
        if half:
            adj = torch.round(2*adj).squeeze()/2 # For conjugation (1.5)
        else:
            adj = torch.round(adj).squeeze() # No conjugation

    return features, adj

def draw_mol(atom_fea_ext, adj):

    features, adj = round_mol(atom_fea_ext, adj)
    
    mol = MolFromGraph(features, adj)
    
    img = MolToImage(mol)
    
    img.save("generated_mol.png")

    return features, adj

def start_from(data):

    N = max_size

    tidx = torch.tril_indices(row=N, col=N, offset=-1)
    
    # Get to correct shape
    atom_fea = torch.zeros(N, n_onehot+1)
    atom_fea[:data.x.shape[0],:n_onehot] = 1*data.x[:,:n_onehot]
    atom_fea[data.x.shape[0]:, n_onehot] = 1
    
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

def MolFromGraph(features, adjacency_matrix):

    atoms = np.array(["H","C","N","O","F"])
    
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(features)):
        atom_type = atoms[(features[i,:n_onehot]==1).numpy()]
        if len(atom_type) > 0:
            a = Chem.Atom(atom_type[0])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx
        else:
            node_to_idx[i] = None

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0 or bond == 0.5:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 1.5:
                bond_type = Chem.rdchem.BondType.AROMATIC
            elif bond == 2 or bond == 2.5:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond >= 3:
                bond_type = Chem.rdchem.BondType.TRIPLE

            if node_to_idx[ix] is not None and node_to_idx[iy] is not None:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol

def prepare_data(data):
    # Get to correct shape

    N = max_size

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

def prepare_data_vector(data):
    # Get to correct shape
    atom_fea = 1*data.x[:,:n_onehot]

    atom_fea = torch.cat([atom_fea, torch.zeros(atom_fea.shape[0],1), torch.matmul(atom_fea[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1)
    
    #N = torch.max(torch.tensor([torch.sum(data.batch==i) for i in range(data.num_graphs)]))
    N = max_size
    
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

def prepare_data_from_features(features, adj):
    # Get to correct shape

    N = features.shape[0]
 
    new_atom_fea = torch.cat([features, torch.matmul(features[:,:n_onehot], torch.tensor([[1.0/8],[4.0/8],[5.0/8],[6.0/8],[7.0/8]]))],dim=1).unsqueeze(0)

    adj = adj.unsqueeze(0)
    
    return new_atom_fea, adj
                
# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nudge(atom_fea, adj):
    atom_fea = atom_fea + torch.randn(*atom_fea.shape)*noise_factor
    adj = adj + torch.randn(*adj.shape)*noise_factor
    return atom_fea, adj

def keep_in(data):
    return len(data.x) <= max_size

def train(qm9):
    """ Training of CGCNN (vectorized) on the qm9 database """
    
    print("Size of database:", len(qm9))
    
    qm9_loader_train = DataLoader(qm9[:n_data], batch_size = batch_size, shuffle = True)
    qm9_loader_valid = DataLoader(qm9[n_data:n_data + n_data//10], batch_size = batch_size, shuffle = True)
    
    std = torch.std(qm9.data.y[:n_data,4])
    mean = torch.mean(qm9.data.y[:n_data,4])
    
    # Initialize network
    model = CGCNN(n_onehot+2,
                     atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                     classification=False).to(device)
    
    if use_pretrained:
        model.load_state_dict(torch.load('model_weights.pth'))
    else:
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        for epoch in range(num_epochs):
    
            model.train()
            
            epoch_loss_train.append(0)
            epoch_scores = []
            epoch_targets = []
            for batch_idx, data in enumerate(qm9_loader_train):
                # Get data to cuda if possible
                data = data.to(device=device)
        
                inputs = prepare_data_vector(data)
    
                inputs = nudge(*inputs) # Make the model more tolerent of non integers
                
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
                scores_valid = model(*prepare_data_vector(data)).squeeze()
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
        
        torch.save(model.state_dict(), 'model_weights.pth')
                
        plt.ioff()
        plt.savefig("progress.png")

    model.eval()
        
    plt.figure()
    
    criterion = nn.MSELoss()
    
    data_train = next(iter(DataLoader(qm9[:n_data], batch_size = n_data, shuffle = False)))
    plt.plot(data_train.y[:,4], model(*prepare_data_vector(data_train)).detach().numpy(), ".")
    
    print("FINAL TRAIN RMSE", criterion(data_train.y[:,4], model(*prepare_data_vector(data_train)).squeeze()))
    
    data_valid = next(iter(DataLoader(qm9[n_data:n_data + n_data//10], batch_size = n_data//10, shuffle = False)))
    plt.plot(data_valid.y[:,4], model(*prepare_data_vector(data_valid)).detach().numpy(),".")
    
    print("FINAL TEST RMSE", criterion(data_valid.y[:,4], model(*prepare_data_vector(data_valid)).squeeze()))
    
    ax = plt.gca()
    
    x = np.linspace(0,18,300)
    ax.fill_between(x, x+1, x-1, color="gray", alpha=0.1)
    
    plt.plot(x, x, color="k", alpha=0.5)
    
    ax.set_aspect("equal","box")

    return model

def initialize(qm9):

    N = max_size
        
    fea_h, adj_vec = start_from(qm9[qmid])
    
    if from_random:
        adj_vec = torch.randn((N*(N-1)//2))
        fea_h = torch.randn((1, N, n_onehot+1))
        pickle.dump((adj_vec, fea_h), open("save_random.pkl","wb"))
    elif from_saved_random:
        adj_vec, fea_h = pickle.load(open("save_random.pkl","rb"))
    else:
        print("Model estimate for random pick:", model(*prepare_data(qm9[qmid])))
    
        init_features, init_adj = prepare_data(qm9[qmid])
    
        mol = MolFromGraph(init_features[0,:,:n_onehot],init_adj.squeeze())
    
        img = MolToImage(mol)
    
        img.save("initial_mol.png")
    
        
    print("INIT ADJ_VEC", adj_vec)
    print("INIT FEA_H", fea_h)
    
    adj_vec.requires_grad = True
    fea_h.requires_grad = True

    return fea_h, adj_vec 


def get_grad_norms(optimizer, params, losses):


    grad_total = [0]*len(params)
    grad2_total = [0]*len(params)

    for loss in losses:
        for i, p in enumerate(params):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad_total[i] += p.grad.data.clone()
            grad2_total[i] += p.grad.data.clone()**2

    grad_total = torch.sum(torch.tensor([torch.mean(g**2) for g in grad_total]))
    grad2_total = torch.sum(torch.tensor([torch.mean(g) for g in grad2_total]))
            
    return grad_total, grad2_total

def get_loss_coeffs(optimizer, params, losses):

    grads = []

    for loss in losses:
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grads.append([p.grad.data.clone() for p in params])

    # Normalize the gradients
    gn = gradient_normalizers(grads, losses, "loss+") # l2, loss, loss+, none  
    
    grads = [[g/gn[i] for g in gs] for i,gs in enumerate(grads)]
    
    sol, min_norm = MinNormSolver.find_min_norm_element(grads)

    return [float(s) for s in sol]

def shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t
    
def inverter(model, fea_h, adj_vec):
    """ Backpropagates all the way to the input to produce a molecule with a specific property """

    # Freezes the trained model
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([adj_vec, fea_h], lr=inv_r)
    
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
    for e in range(n_iter):
        atom_fea_ext, adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec)
        
        score = model(atom_fea_ext, adj)
        
        loss = abs(score - torch.tensor([[target_value]]))

        n_dip = e - dip
        
        losses_floats = [n_dip/100*float(loss), float(constraints), 0*float(integer_fea), 0*float(integer_adj)]

        # # if float(loss) < 0.3:
        # #     losses_floats[0] = 0
        
        # if True: #np.max(losses_floats) - np.min(losses_floats) > 1:
        #     # c = np.array([1,1,1,1])
        #     c = np.array([0,10,0,10])
            
        c = np.array(losses_floats)/np.sum(losses_floats)
        #     g1, g2 = get_grad_norms(optimizer, [fea_h, adj_vec], [c[0]*loss, c[1]*constraints, c[2]*integer_fea, c[3]*integer_adj])
        #     g1 = g1
        #     g2 = g2/g1
            
        # else:
        #     c = get_loss_coeffs(optimizer, [fea_h, adj_vec], [loss, constraints, integer_fea, integer_adj])

        # # if np.max(c) < 0.5 or cycle:
        # #     cycle = True
        # #     c = 1/np.array(losses_floats)
        # #     c = c/np.sum(c)
            
        # # if (g2 > 1000 and total_loss > 0.1) or (cycle and extra < 1000):
        # #     if not cycle:
        # #         cycle = True
        # #         new_c = np.zeros(4)
        # #         new_c[np.argmax(c)] = 1
        # #     extra += 1
        # #     c = get_loss_coeffs(optimizer, [fea_h, adj_vec], [loss, constraints, integer_fea, integer_adj])
        # # else:
        # #     cycle = False
        #     extra = 0

        #c = np.array([1,1,0,1])

        if loss < stop_loss:
            dip = e
            c[0] = lcount/10000
            lcount += 1
        else:
            lcount = 0
        if constraints < stop_chem:
            c[1] = ccount/10000
            cdip = e
            ccount += 1
        else:
            ccount = 0
            
        if constraints < stop_chem and loss < stop_loss:
            break
        
        total_loss = c[0]*loss + c[1]*constraints + c[2]*integer_fea + c[3]*integer_adj


        total_losses.append(float(total_loss))
        losses.append(float(loss))
        constraint_losses.append(float(constraints))
        integer_fea_losses.append(float(integer_fea))
        integer_adj_losses.append(float(integer_adj))

        #g1s.append(float(g1))
        #g2s.append(float(g2))
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # gradient descent or adam step
        optimizer.step()

        # if g2 > 100 and total_loss > 0.1:
        #         with torch.no_grad():
        #             N = max_size
        #             adj_vec += torch.randint(-1,1, ((N*(N-1)//2),))*0.5
        #             fea_h = shuffle(fea_h)
    
        if e%10 == 0:
            print(c)
            #print(float(total_loss), float(loss), float(constraints), float(integer_fea), float(integer_adj), float(score))
            
        if e%1000 == 0:
            plt.plot(total_losses,'k')
            plt.plot(losses,'b')
            plt.plot(constraint_losses,".-", color="r")
            plt.plot(integer_fea_losses, color="orange")
            plt.plot(integer_adj_losses, color="purple")
            plt.plot(g1s, color="pink")
            plt.plot(g2s, color="green")
            plt.pause(0.1)
            draw_mol(atom_fea_ext, adj)
    
    plt.ioff()
    return fea_h, adj_vec

if __name__ == "__main__":

    qm9 = datasets.QM9("qm9_small", pre_filter=keep_in)

    if use_pretrained:
        idx = pickle.load(open("qm9_order.pickle","rb"))
    else:
        if random_split:
            idx = torch.randperm(len(qm9))
        else:
            idx = list(range(len(qm9)))
        pickle.dump(idx, open("qm9_order.pickle","wb"))
    
    qm9 = qm9[idx]
    
    
    model = train(qm9)

    
    fea_h, adj_vec = initialize(qm9)

    init_atom_fea_ext, init_adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec)    
    print("Model estimate for starting point:", model(init_atom_fea_ext, init_adj), constraints, integer_fea, integer_adj)
    
    fea_h, adj_vec = inverter(model, fea_h, adj_vec)
    

    # Printing the result ---------------------------------------------------------------------------
    
    atom_fea_ext, adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec)
    
    print("Final value:", model(atom_fea_ext, adj))
    
    features, adj_round = draw_mol(atom_fea_ext, adj)
    
    atom_fea_ext_r, adj_r = prepare_data_from_features(features, adj_round)    
    atom_fea_ext_r_2, adj_r_2 = prepare_data_from_features(*round_mol(atom_fea_ext, adj, half=True))
    
    print("Final value after rounding:", model(atom_fea_ext_r, adj_r))
    print("Final value after half:", model(atom_fea_ext_r_2, adj_r_2))
    print("Final value after rounding (adj_r):", model(atom_fea_ext, adj_r))
    print("Final value after rounding (fea_r):", model(atom_fea_ext_r, adj))
    
    print("ROUNDING DIFF")
    print("FEA")
    print(torch.max(abs(atom_fea_ext - atom_fea_ext_r)))
    print("ADJ")
    print(torch.max(abs(adj - adj_r)))
    print(abs(atom_fea_ext - atom_fea_ext_r))
    print(abs(adj - adj_r))

    print("STARTING ATOM FEA")
    print(init_atom_fea_ext)
    print("STARTING ADJ")
    print(init_adj)
    print("FINAL ADJ VEC")
    print(adj_vec)
    print("FINAL ATOM FEA")
    print(atom_fea_ext)
    print("FINAL ADJ")
    print(adj)
    print("ROUNDED FINAL ATOM FEA")
    print(features)
    print("ROUNDED FINAL ADJ")
    print(adj_round)
    
    bonds_per_atom = torch.matmul(atom_fea_ext[0,:,:n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0]))
    print("BONDS PER ATOM")
    print(bonds_per_atom)
    print("SUM ADJ")
    print(torch.sum(adj, dim=1))

    r_bonds_per_atom = torch.matmul(features, torch.tensor([1.0,4.0,3.0,2.0,1.0,0.0]))
    print("Rounded BONDS PER ATOM")
    print(r_bonds_per_atom)
    print("Rounded SUM ADJ")
    print(torch.sum(adj_round, dim=1))
    if torch.sum(abs(r_bonds_per_atom - torch.sum(adj_round, dim=1))) > 1e-12:
        print("FINAL STOCHIOMETRY IS WRONG!")
    
    plt.show()
