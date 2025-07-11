import os
import sys
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import one_hot, scatter

import pickle

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class QM9like(InMemoryDataset):
    r"""
        A dataset class to process datasets in the same way it is processed for QM9
        in torch_geometric.

        Args:

                root (str): Root directory where the dataset should be saved.
                transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
                pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
                pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
                raw_name (str, optional): The name of the raw data file to be processed.
                    (default: 'data')
                type_list (list, optional): A list of elements that sets the order of
                the onehot encoding.
                (default: ['H', 'C', 'N', 'O', 'F'])
                return_type_list (bool, optional): If True, the dataset will return a
                dictionary with the types and their corresponding indices and bond
                types. Useful if you want to know possible valances in a dataset.
                If False, it will return a list of indices.
                (default: False)
                atom_class (bool, optional): If True, the dataset will return an
                additional attribute `atom_class` which is a one-hot encoding of the
                atom classes. This is useful for datasets that have atom classes
                (default: False)
        Returns:
                :class:`torch_geometric.data.InMemoryDataset`: The processed dataset.
    """

    raw_url = ""
    raw_url2 = ""
    processed_url = ""

    default_type_list = ['H', 'C', 'N', 'O', 'F']

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, raw_name="data", type_list = default_type_list, return_type_list=False, atom_class=False):
        print(root, transform, pre_transform, pre_filter)
        self.atom_class = atom_class
        self.raw_name = raw_name
        self.return_type_list = return_type_list
        if self.return_type_list:
            self.types = dict()
        else:
            self.types = type_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.types = torch.load("/".join(self.processed_paths[0].split("/")[:-1]) + "/onehot.pt")
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        import rdkit  # noqa
        return [self.raw_name + '.sdf', self.raw_name + '.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)\

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)

        if self.atom_class:
            atom_classes = pickle.load(open(self.raw_paths[2], "rb"))

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        if self.return_type_list:
            for mol in tqdm(suppl):
                for atom in mol.GetAtoms():
                    symb = atom.GetSymbol()
                    nb = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
                    if symb in self.types:
                        if nb not in self.types[symb][1]:
                            self.types[symb][1].append(nb)
                    else:
                        self.types[symb] = [len(self.types), [nb]]

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                if self.return_type_list:
                    type_idx.append(self.types[atom.GetSymbol()][0])
                else:
                    type_idx.append(self.types.index(atom.GetSymbol()))

                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()

            x = torch.cat([x1, x2], dim=-1)

            y = target[i].unsqueeze(0)

            name = mol.GetProp('_Name')

            if self.atom_class:
                atom_class = one_hot(torch.tensor(atom_classes[i]), num_classes=69)

                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=y, name=name, idx=i, atom_class=atom_class)
            else:

                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=y, name=name, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.types, "/".join(self.processed_paths[0].split("/")[:-1]) + "/onehot.pt")
        torch.save(self.collate(data_list), self.processed_paths[0])
