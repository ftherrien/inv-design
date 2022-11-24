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

def generate(target_property, n, output, config=None):
    
    if config is None:
        config = str(Path(__file__).resolve().parents[1]) + "/config.yml"
    
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in config.items()})

    os.makedirs(output, exist_ok=True)
    os.makedirs(output + "/drawings", exist_ok=True)
    os.makedirs(output + "/xyzs", exist_ok=True)
    
    def keep_in(data):
        return len(data.x) <= config.property_model_training.max_size

    if not config.property_model_training.use_pretrained:
        qm9 = datasets.QM9(output + "/dataset", pre_filter=keep_in)
        
        if os.path.isfile(output + "/saved_order.pickle"):
            idx = pickle.load(open(output + "/saved_order.pickle","rb"))
        else:
            if config.property_model_training.random_split:
                idx = torch.randperm(len(qm9))
            else:
                idx = list(range(len(qm9)))
            pickle.dump(idx, open(output + "/saved_order.pickle","wb"))
        
        qm9 = qm9[idx]
    else:
        if config.inverter.start_from in ["random", "saved"]:
            qm9 = None
        else:
            qm9 = datasets.QM9(output + "/dataset", pre_filter=keep_in)
            
    model = train(qm9, config.property_model_training, output)
    
    print("Starting molecule generation loop")

    f = open(output + "/property_value_list.txt","w")

    i=0
    j=0
    while i < n and j < n*config.inverter.max_attempts:
        
        print("------------------------------------")
        print("Molecule %d:"%i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        fea_h, adj_vec = initialize(qm9, config.inverter, output, j, device=device)
        
        init_atom_fea_ext, init_adj, constraints, integer_fea, integer_adj = weights_to_model_inputs(fea_h, adj_vec, config.inverter)    
        print("Model estimate for starting point:", model(init_atom_fea_ext, init_adj), constraints, integer_fea, integer_adj)

        print("Generating molecule with requested property...")
        fea_h, adj_vec = invert(target_property, model, fea_h, adj_vec, config.inverter, output)
        
        # Printing the result ---------------------------------------------------------------------------
        
        atom_fea_ext, adj, _, _, _ = weights_to_model_inputs(fea_h, adj_vec, config.inverter)

        features, adj_round = round_mol(atom_fea_ext, adj, config.inverter.n_onehot)

        r_bonds_per_atom = torch.matmul(features, torch.tensor([1.0,4.0,3.0,2.0,1.0,0.0], device=device))
        
        if torch.sum(abs(r_bonds_per_atom - torch.sum(adj_round, dim=1))) > 1e-12:
            print("Generated molecule is not stochiometric")
            j+=1
            continue
        else:
            print("Generation successful")

        print("Final property value:", model(atom_fea_ext, adj))
            
        features, adj_round, smiles = draw_mol(atom_fea_ext, adj, config.inverter.n_onehot, output, index=i, embed=True)
        
        atom_fea_ext_r, adj_r = prepare_data_from_features(features, adj_round, config.inverter.n_onehot)    

        val = model(atom_fea_ext_r, adj_r)
        print("Final property value after rounding:", val)
        print("Final property value after rounding (adj_r):", model(atom_fea_ext, adj_r))
        print("Final property value after rounding (fea_r):", model(atom_fea_ext_r, adj))
        
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

        print("Generated Molecule SMILES:")
        print(smiles)
        
        # Print value to file
        print(i, smiles, float(val),file=f)

        i+=1
        j+=1
        
    plt.show()
