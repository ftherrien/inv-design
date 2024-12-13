# Generated molecules DFT dataset

When gathering data for the energy gap task, we performed 2410 DFT calculations with ORCA B3LYP/6-31G(2df,2p). These calculations were done on 2017 unique molecular graphs (conformation may differ),
1617 of which were new molecules that cannot be found in QM9.

`data.csv` contains the following fields:

- `mu`: Dipole moment in Debye,
- `E_L`: Lowest unoccupied molecular orbital energy in eV,
- `E_H`: Highest occupied molecular orbital in eV,
- `gap`: HOMO-LUMO energy gap in eV
- `method`: Generation method: algorithm-proxy-target
- `qm9_id`: QM9 ID number if available,
- `smiles_init`: Initial smiles string before relaxation (if available)
- `smiles_final`: Final smiles string after relaxation.


The `xyzs` directory contains the corresponding DFT-relaxed xyz files containing the full 3D information for each molecule.  