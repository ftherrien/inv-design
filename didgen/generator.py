from .train import train, add_extra_features
from .inverter import invert, weights_to_model_inputs, initialize
from .utils import round_mol, draw_mol
from .config import default_config_dict

from types import SimpleNamespace
import yaml
import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
import itertools
from copy import deepcopy
from numpy.random import choice
from bayes_opt import BayesianOptimization
from rdkit import Chem

torch.set_printoptions(sci_mode=False,linewidth = 300)

def get_extra_features_matrix(type_list, extra_features, device):
    """From simple adj and feature matrix add atoms info and adjust dimensions from model input"""
    
    pt = Chem.GetPeriodicTable()
    
    extra_fea_matrix = []
    
    if "atomic_weight" in extra_features:
        extra_fea_matrix.append([pt.GetAtomicWeight(t) for t in type_list])
    if "atomic_number" in extra_features:
        extra_fea_matrix.append([pt.GetAtomicNumber(t) for t in type_list])
    if "n_valence" in extra_features:
        extra_fea_matrix.append([pt.GetNOuterElecs(t)/8 for t in type_list])

    return torch.tensor(extra_fea_matrix, device=device).T

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

    tmpconfig = deepcopy(config)
    
    tmpconfig.starting_size = config.max_size
    tmpconfig.start_from = "random"

    for j in range(config.mini_hpo.n_starts):
        initialize(None, tmpconfig, output, j, device=device)
    

def test_params(model, config, output, device, *params):

    print("PARAMS:", params, flush=True)
    
    tmpconfig = deepcopy(config)

    tmpconfig.inv_r = params[0]
    tmpconfig.l_loss = params[1]
    tmpconfig.l_const = params[2]
    tmpconfig.starting_size = params[3]


    tmpconfig.start_from = "saved"
    tmpconfig.n_iter = config.mini_hpo.n_iter
    tmpconfig.show_losses = False
    tmpconfig.max_attemps = 1
    tmpconfig.stop_chem = config.mini_hpo.stop_chem
    tmpconfig.stop_loss = config.mini_hpo.stop_loss

    score = 0
    avg_n_iter = 0
    for j in range(config.mini_hpo.n_starts):
    
        fea_h, adj_vec = initialize(None, tmpconfig, output, j, device=device)
        
        fea_h, adj_vec, n_iter_final = invert(model, fea_h, adj_vec, tmpconfig, output)
        
        atom_fea_ext, adj = weights_to_model_inputs(fea_h, adj_vec, tmpconfig)
        
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
        avg_n_iter = (config.mini_hpo.n_iter-1)
        
    return score + 1 - avg_n_iter/(config.mini_hpo.n_iter-1)

    
def bayesian_hpo(device, output, model, config):
    
    pbounds = {"log_inv_r": tuple(torch.log(torch.tensor(config.inv_r))),
              "log_l_loss": tuple(torch.log(torch.tensor(config.l_loss))),
              "log_l_const": tuple(torch.log(torch.tensor(config.l_const))),
              "starting_size": tuple(torch.tensor(config.starting_size))}

    print(pbounds)
    
    init_test_params(config, output, device)

    def black_box(log_inv_r, log_l_loss, log_l_const, starting_size):
        starting_size = int(torch.round(torch.tensor(starting_size)))
        inv_r = torch.exp(torch.tensor(log_inv_r))
        l_loss = torch.exp(torch.tensor(log_l_loss))
        l_const = torch.exp(torch.tensor(log_l_const))
        return test_params(model, config, output, device, inv_r, l_loss, l_const, starting_size)

    optimizer = BayesianOptimization(f=black_box,
                                     pbounds=pbounds,
                                     random_state=1)

    import numpy as np
    with np.errstate(divide='ignore',invalid='ignore'):
        optimizer.maximize(init_points=2, n_iter=config.mini_hpo.n_comb)
    
    outconfig = deepcopy(config)

    print(optimizer.max)

    outconfig.inv_r         = float(torch.exp(torch.tensor(optimizer.max["params"]["log_inv_r"])))
    outconfig.l_loss        = float(torch.exp(torch.tensor(optimizer.max["params"]["log_l_loss"])))
    outconfig.l_const       = float(torch.exp(torch.tensor(optimizer.max["params"]["log_l_const"])))
    outconfig.starting_size = int(torch.round(torch.tensor(optimizer.max["params"]["starting_size"])))

    print("CONFIG", outconfig)
        
    return outconfig

def random_hpo(device, output, model, config):
    
    list_params = random_product(config.mini_hpo.n_comb, config.inv_r, config.l_loss, config.l_const, config.starting_size)

    init_test_params(config, output, device)
    
    scores = torch.tensor([test_params(model, config, output, device, *params) for params in list_params])

    print("SCORES", scores)
    
    id_max = torch.argmax(scores)

    outconfig = deepcopy(config)

    outconfig.inv_r         = list_params[id_max][0]
    outconfig.l_loss        = list_params[id_max][1]
    outconfig.l_const       = list_params[id_max][2]
    outconfig.starting_size = list_params[id_max][3]

    print("Best config", outconfig)
        
    return outconfig


def to_SimpleNamespace(conf_dict, default=None):
    
    if default is not None:
        conf_dict = {**default, **conf_dict}

    return SimpleNamespace(**{k:(to_SimpleNamespace(v) if type(v) is dict else v) for k,v in conf_dict.items()})

def from_SimpleNamespace(namespace):

    conf_dict = deepcopy(vars(namespace))
    
    for k,v in conf_dict.items():
        if type(v) is SimpleNamespace:
            conf_dict[k] = from_SimpleNamespace(conf_dict[k])

    return conf_dict    

def generate(n, output, config=None, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config is None:
        config_dict = {}
    elif type(config) is dict:
        config_dict = config
    else:
        with open(config) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

    all_config_dict = {**default_config_dict, **config_dict}
            
    config = to_SimpleNamespace(all_config_dict)

    config.bonding = torch.tensor(config.bonding, device=device)

    config.bonding_mask = [(config.bonding == i) for i in set(config.bonding.tolist()) if i > 0] 
    
    config._extra_fea_matrix = get_extra_features_matrix(config.type_list, config.extra_features, device)

    os.makedirs(output, exist_ok=True)
    os.makedirs(output + "/drawings", exist_ok=True)
    os.makedirs(output + "/xyzs", exist_ok=True)
            
    model = train(config, output)
    
    print("Starting molecule generation loop", flush=True)

    f = open(output + "/property_value_list.txt","w")
    
    if config.mini_hpo.method == "random":
        config = random_hpo(device, output, model, config)

        config.mini_hpo.method = False
        
        yaml.dump(from_SimpleNamespace(config), open(output + "/config_optim.yml","w"))

    elif config.mini_hpo.method == "bayesian":

        config = bayesian_hpo(device, output, model, config)

        config.mini_hpo.method = False
        
        yaml.dump(from_SimpleNamespace(config), open(output + "/config_optim.yml","w"))

    output_list = [] 
    i=0
    j=0
    while i < n and j < n*config.max_attempts:
        
        print("------------------------------------")
        print("Molecule %d:"%i)
        
        fea_h, adj_vec, config = initialize(config, output, j, device=device)

        init_atom_fea_ext, init_adj = weights_to_model_inputs(fea_h, adj_vec, config)
        
        print("Model estimate for starting point:", model(init_atom_fea_ext, init_adj))

        print("Generating molecule with requested property...")
        fea_h, adj_vec, n_iter_final = invert(model, fea_h, adj_vec, config, output)
        
        # Printing the result ---------------------------------------------------------------------------
        
        atom_fea_ext, adj = weights_to_model_inputs(fea_h, adj_vec, config)

        features, adj_round = round_mol(atom_fea_ext, adj, len(config.type_list))

        r_bonds_per_atom = torch.matmul(features, torch.cat([config.bonding, torch.zeros(1, device=features.device)]))

        # Number of components in graph

        L = torch.diag(torch.sum((adj_round != 0),axis=0)) - (adj_round != 0)*1

        try:
            n_comp = int(torch.sum(abs(torch.linalg.eigh(L.float())[0]) < 1e-5)) - int(torch.sum(features[:,len(config.type_list)]))
        except:
            n_comp = 2
            
        if torch.sum(abs(r_bonds_per_atom - torch.sum(adj_round, dim=1))) > 1e-12 or n_iter_final == config.n_iter - 1:
            print("Generated molecule is not stochiometric")
            j+=1
            continue
        else:
            print("Generation successful")

        print("Final property value:", model(atom_fea_ext, adj))

        features, adj_round = round_mol(atom_fea_ext, adj, len(config.type_list))

        atom_fea_ext_r = add_extra_features(features, config._extra_fea_matrix).unsqueeze(0)

        adj_r = adj_round.unsqueeze(0)
        
        val = model(atom_fea_ext_r, adj_r)
        print("Final property value after rounding:", val)

        print("Final property value after rounding (adj_r):", model(atom_fea_ext, adj_r))
        print("Final property value after rounding (fea_r):", model(atom_fea_ext_r, adj))
        
        if torch.sum(abs(val - torch.tensor([config.target], device=val.device))) > config.stop_loss:
            print("Target not actually reached!")
            continue
        
        features, adj_round, smiles = draw_mol(atom_fea_ext, adj, config.type_list, output, index=i, embed=config.embed, text=" ".join(["%f"]*(len(val[0])))%tuple(val[0].detach().tolist()), color=(0,255,0))
        
        print("Generated Molecule SMILES:")
        print(smiles)

        print("Number of atoms of each type:")
        print(torch.sum(features,dim=0))

        output_list.append({"features": atom_fea_ext, "adjacency": adj, "features_round": features, "adjacency_round": adj_round, "n_iter":n_iter_final, "smiles":smiles, "property":val[0].detach().tolist()})
        
        # Print value to file
        print(i, n_iter_final, smiles, " ".join(["%f"]*(len(val[0])))%tuple(val[0].detach().tolist()),file=f)

        i+=1
        j+=1

    f.close()
    return output_list
