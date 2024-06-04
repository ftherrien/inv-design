# DIDGen Parameters

Here is a list of parameters for DIDgen. They can be input as a single yaml file and as a python dictionary in the python API. You only need to specify the parameters that you want to change from default. The full list of parameters and their default value can be found in [didgen/config.py](https://github.com/ftherrien/inv-design/blob/master/didgen/config.py).

## Training and datasets
|Parameter|Type|Default|Description|
|---|---|---|---|
|`property`|`list` of `int`|[4]|Index of the property to optimize in the dataset features. Multiple properties can be selected at once. e.g., 4 is the index of the HOMO-LUMO gap property in [PyG QM9](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.QM9.html)|
|`type_list`|`list` of `str`|['H', 'C', 'N', 'O', 'F']|Atomic ordering in the dataset one-hot encoding|
|`max_size`|`int`|25|Maximum molecular size to train on and to generate (including hydrogen atoms)|
|`n_data`|`int`|1000|Size of the dataset to use, if larger than the size of the dataset, all the data is used|
|`datasets`|`list` of `str`|['qm9']|Datasets to train on. 'qm9' will download [PyG QM9](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.QM9.html) otherwise DIDgen will look in `OUTDIR/dataset_name` for a `QM9like` dataset (more on this coming soon, including scripts to use your own datasets)|
|`num_epochs`|`int`|400|Number of epochs|
|`batch_size`|`int`|30|Batch size|
|`learning_rate`|`float`|0.001|Learning rate (Adam optimizer)|
|`noise_factor`|`float`|0.05|Random noises added to input vectors to make the model less sensitive to non-integers (often not necessary)|
|`use_pretrained`|`bool`|False|Whether to reuse weights from previous run, useful when generating molecules without having to retrain. Weights are saved in `OUTDIR/model_weights.pth`.

## Model
|Parameter|Type|Default|Description|
|---|---|---|---|
|`model`|`str`|SimpleNet| Model to train and generate with. Current options are: "SimpleNet", "CrippenNet" and "CGCNN"|
|`layer_list`|`list` of `int`|[128]|List and size of MLP layers after the convolutions. e.g. [128,128,50] would be 3 layer MLP. Only available for "SimpleNet" and "CrippenNet"|
|`atom_fea_len`|`int`|20|Size of the atomic embedding before and during the convolutions|
|`n_conv`|`int`|3|Number of convolutions|
|`weight_decay`|`int`|0|Weight decay as implemented in [Pytorch Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)|
|`dropout`|`int`|0|Dropout rate before each linear layer. Only available for "SimpleNet" and "CrippenNet"|
|`show_train`|`bool`|False|Whether to display and plot training statistics.|
|`atom_class`|`bool`|False|Whether to train on atomic classification. Only available for CrippenNet. *Note that CrippenNet can be trained as a repressor.*|

## Generation
|Parameter|Type|Default|Description|
|---|---|---|---|
|`target`|`list` of `float`|[4.1]|Target value of the property. e.g. An energy gap of 4.1 eV|
|`proportions`|`list` of `float`|[0.411, 0.3026, 0.0602, 0.0735, 0.0015]|Target proportion between different elements following `type_list`. This is set as loose criteria (see `stop_prop`)|
|`stop_loss`|`float`|0.3|Stopping criteria for the property target. DIDgen stops when the value is within `stop_loss` of `target`|
|`stop_prop`|`int`|100|Stopping criteria for element proportions. Set such that criteria is always met. Could be useful to generate isomers.|
|`bonding`|`list` of `int`|[1, 4, 3, 2, 1]|Valence of each element in `type_list`|
|`min_size`|`int`|5|Minimum size of generated molecules|
|`start_from`|`str`|!|Molecular representation to start from. Options are: "random" to start from random noise, "saved" to reuse the previous starting point, "!" to start from a random entry in the dataset and `id` to start from a molecule with index `id` in the dataset.|
|`starting_size`|`int`|25|Maximum starting size when starting from random noise|
|`n_iter`|`int`|300|Maximum number of iterations to reach the target|
|`inv_r`|`float`|0.01|Rate of gradient descent (Adam optimizer)|
|`l_loss`|`float`|0.01|Relative importance of the target property loss (see equation 2 in the manuscript)|
|`l_prop`|`int`|100|Relative importance of the elemental proportion loss|
|`show_losses`|`bool`|False|Whether to display and plot optimization statistics (slow)|
|`max_attempts`|`int`|3|Maximum number of attempts (starts) to reach the target|
|`bond_multiplier`|`float`|0.3|Magnitude of the random noise when using `start_from; "random"`. Larger magnitude creates more bonds.|
|`embed`|`bool`|True|Whether to find a conformer (3D atomic positions) for the generated molecule|
