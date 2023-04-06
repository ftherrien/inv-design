from .train import train, prepare_data_from_features
from .inverter import invert, weights_to_model_inputs, initialize
from .utils import round_mol, draw_mol

from types import SimpleNamespace
import yaml
import torch
from torch_geometric import datasets
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
import itertools
from copy import deepcopy
from numpy.random import choice
from bayes_opt import BayesianOptimization


torch.set_printoptions(sci_mode=False,linewidth = 300)

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def random_product(n, *args):
    sizes = torch.tensor([len(arg) for arg in args])
    Ns = choice(torch.prod(sizes), n, replace=False)
    combs = []
    for N in Ns:
        comb = []
        for i, arg in enumerate(args):
            comb.append(arg[N//int(torch.prod(sizes[i+1:]))])
            N = N%int(torch.prod(sizes[i+1:]))
        combs.append(comb)

    return combs

def init_test_params(config, output, device):

    tmpconfig = deepcopy(config.inverter)
    
    tmpconfig.starting_size = config.inverter.max_size
    tmpconfig.start_from = "random"

    for j in range(config.inverter.mini_hpo.n_starts):
        initialize(None, tmpconfig, output, j, device=device)
    


def test_params(target_property, model, config, output, device, *params):

    print("PARAMS:", params)
    
    tmpconfig = deepcopy(config.inverter)

    tmpconfig.inv_r = params[0]
    tmpconfig.l_loss = params[1]
    tmpconfig.l_const = params[2]
    tmpconfig.starting_size = params[3]


    tmpconfig.start_from = "saved"
    tmpconfig.n_iter = config.inverter.mini_hpo.n_iter
    tmpconfig.show_losses = False
    tmpconfig.max_attemps = 1
    tmpconfig.stop_chem = config.inverter.mini_hpo.stop_chem
    tmpconfig.stop_loss = config.inverter.mini_hpo.stop_loss

    score = 0
    avg_n_iter = 0
    for j in range(config.inverter.mini_hpo.n_starts):
    
        fea_h, adj_vec = initialize(None, tmpconfig, output, j, device=device)
        
        fea_h, adj_vec, n_iter_final = invert(target_property, model, fea_h, adj_vec, tmpconfig, output)
        
        atom_fea_ext, adj, _, _, _ = weights_to_model_inputs(fea_h, adj_vec, tmpconfig)
        
        features, adj_round = round_mol(atom_fea_ext, adj, tmpconfig.n_onehot)
        
        r_bonds_per_atom = torch.matmul(features, torch.tensor([1.0,4.0,3.0,2.0,1.0,0.0], device=device))
        
        # Number of components in graph
        
        L = torch.diag(torch.sum((adj_round != 0),axis=0)) - (adj_round != 0)*1
        
        n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - int(torch.sum(features[:,tmpconfig.n_onehot]))
        
        if n_iter_final < tmpconfig.n_iter - 1 and n_comp < 2:
            score += 1
            avg_n_iter += n_iter_final

    if score > 0:
        avg_n_iter = avg_n_iter/score
    else:
        avg_n_iter = (config.inverter.mini_hpo.n_iter-1)
        
    return score + 1 - avg_n_iter/(config.inverter.mini_hpo.n_iter-1)

    
def bayesian_hpo(device, output, target_property, model, config):
    
    pbounds = {"log_inv_r": tuple(torch.log(torch.tensor(config.inverter.inv_r))),
              "log_l_loss": tuple(torch.log(torch.tensor(config.inverter.l_loss))),
              "log_l_const": tuple(torch.log(torch.tensor(config.inverter.l_const))),
              "starting_size": tuple(torch.tensor(config.inverter.starting_size))}

    print(pbounds)
    
    init_test_params(config, output, device)

    def black_box(log_inv_r, log_l_loss, log_l_const, starting_size):
        starting_size = int(torch.round(torch.tensor(starting_size)))
        inv_r = torch.exp(torch.tensor(log_inv_r))
        l_loss = torch.exp(torch.tensor(log_l_loss))
        l_const = torch.exp(torch.tensor(log_l_const))
        return test_params(target_property, model, config, output, device, inv_r, l_loss, l_const, starting_size)

    optimizer = BayesianOptimization(f=black_box,
                                     pbounds=pbounds,
                                     random_state=1)

    import numpy as np
    with np.errstate(divide='ignore',invalid='ignore'):
        optimizer.maximize(init_points=2, n_iter=config.inverter.mini_hpo.n_comb)
    
    outconfig = deepcopy(config)

    print(optimizer.max)

    outconfig.inverter.inv_r         = float(torch.exp(torch.tensor(optimizer.max["params"]["log_inv_r"])))
    outconfig.inverter.l_loss        = float(torch.exp(torch.tensor(optimizer.max["params"]["log_l_loss"])))
    outconfig.inverter.l_const       = float(torch.exp(torch.tensor(optimizer.max["params"]["log_l_const"])))
    outconfig.inverter.starting_size = int(torch.round(torch.tensor(optimizer.max["params"]["starting_size"])))

    print("CONFIG", outconfig.inverter)
        
    return outconfig

def random_hpo(device, output, target_property, model, config):
    
    list_params = random_product(config.inverter.mini_hpo.n_comb, config.inverter.inv_r, config.inverter.l_loss, config.inverter.l_const, config.inverter.starting_size)

    init_test_params(config, output, device)
    
    scores = torch.tensor([test_params(target_property, model, config, output, device, *params) for params in list_params])

    print("SCORES", scores)
    
    id_max = torch.argmax(scores)

    outconfig = deepcopy(config)

    outconfig.inverter.inv_r         = list_params[id_max][0]
    outconfig.inverter.l_loss        = list_params[id_max][1]
    outconfig.inverter.l_const       = list_params[id_max][2]
    outconfig.inverter.starting_size = list_params[id_max][3]

    print("Best config", outconfig.inverter)
        
    return outconfig


def to_SimpleNamespace(conf_dict):

    return SimpleNamespace(**{k:(to_SimpleNamespace(v) if type(v) is dict else v) for k,v in conf_dict.items()})

def from_SimpleNamespace(namespace):

    conf_dict = deepcopy(vars(namespace))
    
    for k,v in conf_dict.items():
        if type(v) is SimpleNamespace:
            conf_dict[k] = from_SimpleNamespace(conf_dict[k])

    return conf_dict    

def generate(target_property, n, output, config=None):
    
    if config is None:
        config = str(Path(__file__).resolve().parents[1]) + "/config.yml"
    
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = to_SimpleNamespace(config)
    
    os.makedirs(output, exist_ok=True)
    os.makedirs(output + "/drawings", exist_ok=True)
    os.makedirs(output + "/xyzs", exist_ok=True)
    
    def keep_in(data):
        return len(data.x) <= config.property_model_training.max_size

    if not config.property_model_training.use_pretrained:
        qm9 = datasets.QM9(output + "/dataset", pre_filter=keep_in)
    else:
        if config.inverter.start_from in ["random", "saved"]:
            qm9 = None
        else:
            qm9 = datasets.QM9(output + "/dataset", pre_filter=keep_in)
            
    model = train(qm9, config.property_model_training, output)
    
    print("Starting molecule generation loop")

    f = open(output + "/property_value_list.txt","w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.inverter.mini_hpo.method == "random":
        config = random_hpo(device, output, target_property, model, config)

        config.inverter.mini_hpo.method = False
        
        yaml.dump(from_SimpleNamespace(config), open(output + "/config_optim.yml","w"))

    elif config.inverter.mini_hpo.method == "bayesian":

        config = bayesian_hpo(device, output, target_property, model, config)

        config.inverter.mini_hpo.method = False
        
        yaml.dump(from_SimpleNamespace(config), open(output + "/config_optim.yml","w"))

        
    i=0
    j=0
    while i < n and j < n*config.inverter.max_attempts:
        
        print("------------------------------------")
        print("Molecule %d:"%i)
        
        fea_h, adj_vec = initialize(qm9, config.inverter, output, j, device=device)
        
        init_atom_fea_ext, init_adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec, config.inverter)    
        print("Model estimate for starting point:", model(init_atom_fea_ext, init_adj), constraints, integer_fea, integer_adj)

        print("Generating molecule with requested property...")
        fea_h, adj_vec, _ = invert(target_property, model, fea_h, adj_vec, config.inverter, output)
        
        # Printing the result ---------------------------------------------------------------------------
        
        atom_fea_ext, adj, _, _, _ = weights_to_model_inputs(fea_h, adj_vec, config.inverter)

        features, adj_round = round_mol(atom_fea_ext, adj, config.inverter.n_onehot)

        r_bonds_per_atom = torch.matmul(features, torch.tensor([1.0,4.0,3.0,2.0,1.0,0.0], device=device))

        # Number of components in graph

        L = torch.diag(torch.sum((adj_round != 0),axis=0)) - (adj_round != 0)*1

        n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - int(torch.sum(features[:,config.inverter.n_onehot]))
        
        if torch.sum(abs(r_bonds_per_atom - torch.sum(adj_round, dim=1))) > 1e-12:
            print("Generated molecule is not stochiometric")
            j+=1
            continue
        else:
            print("Generation successful")

        print("Final property value:", model(atom_fea_ext, adj))

        print(adj)
        print(atom_fea_ext)

        features, adj_round = round_mol(atom_fea_ext, adj, config.inverter.n_onehot)
        
        atom_fea_ext_r, adj_r = prepare_data_from_features(features, adj_round, config.inverter.n_onehot)    

        val = model(atom_fea_ext_r, adj_r)
        print("Final property value after rounding:", val)
        print("Final property value after rounding (adj_r):", model(atom_fea_ext, adj_r))
        print("Final property value after rounding (fea_r):", model(atom_fea_ext_r, adj))

        features, adj_round, smiles = draw_mol(atom_fea_ext, adj, config.inverter.n_onehot, output, index=i, embed=True, text="%f"%(val), color=(0,255,0))
        
        print("ROUNDING DIFF")
        print("FEA")
        print(torch.max(abs(atom_fea_ext - atom_fea_ext_r)))
        print("ADJ")
        print(torch.max(abs(adj - adj_r)))
        # print(abs(atom_fea_ext - atom_fea_ext_r))
        # print(abs(adj - adj_r))
        
        # print("STARTING ATOM FEA")
        # print(init_atom_fea_ext)
        # print("STARTING ADJ")
        # print(init_adj)
        # print("FINAL ADJ VEC")
        # print(adj_vec)
        # print("FINAL ATOM FEA")
        # print(atom_fea_ext)
        # print("FINAL ADJ")
        # print(adj)
        print("ROUNDED FINAL ATOM FEA")
        print(features)
        print("ROUNDED FINAL ADJ")
        print(adj_round)
        
        bonds_per_atom = torch.matmul(atom_fea_ext[0,:,:config.inverter.n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0], device=device))
        print("BONDS PER ATOM")
        print(bonds_per_atom)
        print("SUM ADJ")
        print(torch.sum(adj, dim=1))
        
        print("Rounded BONDS PER ATOM")
        print(r_bonds_per_atom)
        print("Rounded SUM ADJ")
        print(torch.sum(adj_round, dim=1))

        print("Number of components (molecules) in generated graph:", n_comp)
        print("Generated Molecule SMILES:")
        print(smiles)
        
        # Print value to file
        print(i, n_comp, smiles, float(val),file=f)

        i+=1
        j+=1
        
    plt.show()
