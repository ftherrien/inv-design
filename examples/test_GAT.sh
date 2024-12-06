#!/bin/bash

name=tests_GAT

mkdir -p out_${name}

cat <<EOF > out_${name}/config.yml
property: [4]
target: [5]

max_size: 25
# type_list: [C, S, Si, O, H, N, Se] #cep
# bonding: [4, 2,  4, 2, 1, 3, 2] #cep
type_list: [H, C, N, O, F] #qm9
bonding: [1, 4, 3, 2, 1] #qm9
extra_features: [n_valence] # atomic_weight, atomic_number, n_valence
#proportions: [0.3, 0.01, 0.01, 0.05, 0.4, 0.05, 0.01] #cep
proportions: [0.4110, 0.3026, 0.0602, 0.0735, 0.0015] #qm9

n_data: 1000
datasets: [qm9]
num_epochs: 200
batch_size: 30
learning_rate: 0.001
noise_factor: 0.0
transfer_learn: False
use_pretrained: False
model: "GAT"
pooling: "mean"
layer_list: [128]
shuffle: False
n_conv: 3
atom_fea_len: 10
dropout: 0
weight_decay: 0
JK: "last"
batch_norm: True
show_train: True

min_size: 2 
starting_size: 25
start_from: "random"
n_iter: 300
mini_hpo:
    method: False
    n_comb: 30
    n_starts: 10
    n_iter: 5000
    stop_chem: 0.3
    stop_loss: 1
inv_r: 0.01
l_loss: 1
l_const: 100
l_prop: 100
stop_loss: 0.3
stop_prop: 100
adj_eps: 0.1
show_losses: True
max_attempts: 10
rounding: step
bond_multiplier: 0.4

EOF

didgenerate -n 10 -c out_${name}/config.yml -o out_${name}

# didgenerate -p 3 -c config.yml -o out_20/
