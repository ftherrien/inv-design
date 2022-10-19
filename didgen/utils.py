from rdkit.Chem.rdmolfiles import SDMolSupplier, MolToXYZFile
from rdkit.Chem.AllChem import EmbedMolecule, UFFOptimizeMolecule
from rdkit.Chem.Draw import MolToImage
import rdkit.Chem as Chem
import torch
import numpy as np
import pickle

def round_mol(atom_fea_ext, adj, n_onehot, smooth=False, half=False):

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

def draw_mol(atom_fea_ext, adj, n_onehot, output, index=0, embed=False):

    features, adj = round_mol(atom_fea_ext, adj, n_onehot)
    
    mol = MolFromGraph(features, adj, n_onehot)

    pickle.dump(mol,open(output+"/xyzs/generated_mol_%d.pickle"%(index),"wb"))
    
    img = MolToImage(mol)
    
    img.save(output+"/drawings/generated_mol_%d.png"%(index))

    if embed:
        Chem.SanitizeMol(mol)
        idc = EmbedMolecule(mol, 1000)
        if idc != -1:
            UFFOptimizeMolecule(mol)
            MolToXYZFile(mol, output+"/xyzs/generated_mol_%d.xyz"%(index),idc)
        else:
            print("Embeding failed!")
            
    return features, adj

def MolFromGraph(features, adjacency_matrix, n_onehot):

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
