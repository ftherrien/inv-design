from .train import train, prepare_data_from_features
from .inverter import invert, weights_to_model_inputs, initialize
from .utils import round_mol, draw_mol

from types import SimpleNamespace
import yaml
import torch
from torch_geometric import datasets
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import pickle

def generate(target_property, n, output, config):
    
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in config.items()})

    os.makedirs(output, exist_ok=True)
    
    def keep_in(data):
        return len(data.x) <= config.property_model_training.max_size
    
    qm9 = datasets.QM9(output + "/dataset", pre_filter=keep_in)

    if config.property_model_training.use_pretrained and os.path.isfile(output + "/saved_order.pickle"):
        idx = pickle.load(open(output + "/saved_order.pickle","rb"))
    else:
        if config.property_model_training.random_split:
            idx = torch.randperm(len(qm9))
        else:
            idx = list(range(len(qm9)))
        pickle.dump(idx, open(output + "/saved_order.pickle","wb"))
    
    qm9 = qm9[idx]
    
    model = train(qm9, config.property_model_training, output)

    fea_h, adj_vec = initialize(qm9, config.inverter, output)

    init_atom_fea_ext, init_adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec, config.inverter)    
    print("Model estimate for starting point:", model(init_atom_fea_ext, init_adj), constraints, integer_fea, integer_adj)
    
    fea_h, adj_vec = invert(target_property, model, fea_h, adj_vec, config.inverter, output)

    # Printing the result ---------------------------------------------------------------------------
    
    atom_fea_ext, adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec, config.inverter)
    
    print("Final value:", model(atom_fea_ext, adj))
    
    features, adj_round = draw_mol(atom_fea_ext, adj, config.inverter.n_onehot, output)
    
    atom_fea_ext_r, adj_r = prepare_data_from_features(features, adj_round, config.inverter.n_onehot)    
    atom_fea_ext_r_2, adj_r_2 = prepare_data_from_features(*round_mol(atom_fea_ext, adj, config.inverter.n_onehot, half=True), config.inverter.n_onehot)
    
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
    
    bonds_per_atom = torch.matmul(atom_fea_ext[0,:,:config.inverter.n_onehot], torch.tensor([1.0,4.0,3.0,2.0,1.0]))
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
