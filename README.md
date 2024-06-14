# Using GNN property predictors as molecule generators (DIDgen)

This is the repository for the paper: [*Using GNN property predictors as molecule generators*](https://arxiv.org/abs/2406.03278).

You can use DIDgen (**D**irect **I**nverse **D**esign **gen**erator) to generate diverse molecules with a specific property by *inverting* a GNN that predicts that property.

<img src="https://github.com/ftherrien/inv-design/blob/master/anim89.gif" width="200" height="200"><img src="https://github.com/ftherrien/inv-design/blob/master/anim32.gif" width="200" height="200"><img src="https://github.com/ftherrien/inv-design/blob/master/anim111.gif" width="200" height="200">

## Install

```
pip install git+https://github.com/ftherrien/inv-design
```

## Usage

If you have trouble using or installing DIDgen please [create and issue](https://github.com/ftherrien/inv-design/issues/new) or [ask a question](https://github.com/ftherrien/inv-design/discussions/new?category=q-a). I will be happy to help! 

### As a command-line interface (cli)

```
didgenerate [-h] [-n N] [-c CONFIG] [-o OUTDIR]
```

The results are organized in `OUTDIR` as such

```
OUTDIR
├── drawings
│   └── generated_mol_0.png     # An image(s) of the generated graph(s)
├── final_performance_data.pkl
├── final_performance.png
├── initial_mol.png
├── model_weights.pth
├── property_value_list.txt     # A list of smiles strings and corresponding predicted property
├── qm9/
└── xyzs/
    ├── generated_mol_0.pickle  # A RDKit mol object
    └── generated_mol_0.xyz     # A molecular conformer with 3D positions
```

### As a Python API

```python
from didgen import generate

out = generate(number_of_samples, outdir, config_dict)
```

This creates the same output directory as the cli. `out` is a list of python dictionaries containing the generated graphs, their corresponding smiles and the predicted property.

### Parameters

You can find a list of parameters and their description in [the documentation](https://github.com/ftherrien/inv-design/blob/master/docs/parameters.md).

## Generate molecules online using Colab

[Train a GNN to predict the energy gap on a subset of QM9 and generate a molecule with an energy gap of 4.1 eV](https://colab.research.google.com/github/ftherrien/inv-design/blob/master/didgenerate.ipynb)

## Citation

```
@misc{therrien2024using,
      title={Using GNN property predictors as molecule generators}, 
      author={Félix Therrien and Edward H. Sargent and Oleksandr Voznyy},
      year={2024},
      eprint={2406.03278},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```