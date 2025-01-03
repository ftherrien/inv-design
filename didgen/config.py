default_config_dict = {'property': [4],
                       'target': [4.1],
                       'max_size': 25,
                       'type_list': ['H', 'C', 'N', 'O', 'F'],
                       'bonding': [1, 4, 3, 2, 1],
                       'extra_features': ['n_valence'],
                       'proportions': [0.4110, 0.3026, 0.0602, 0.0735, 0.0015],
                       'n_data': 1000,
                       'datasets': ['qm9'],
                       'num_epochs': 400,
                       'batch_size': 30,
                       'learning_rate': 0.001,
                       'noise_factor': 0.05,
                       'transfer_learn': False,
                       'use_pretrained': False,
                       'model': 'SimpleNet',
                       'pooling': 'smartest',
                       'layer_list': [128],
                       'atom_fea_len': 20,
                       'shuffle': False,
                       'n_conv': 3,
                       'batch_norm': True,
                       'weight_decay': 0,
                       'dropout': 0,
                       'show_train': False,
                       'atom_class': False,
                       'alpha': 0,
                       'n_rand_samples': 6700,
                       'min_size': 5,
                       'starting_size': 25,
                       'start_from': '!',
                       'n_iter': 300,
                       'mini_hpo': {'method': False,
                                    'n_comb': 30,
                                    'n_starts': 10,
                                    'n_iter': 5000,
                                    'stop_chem': 0.3,
                                    'stop_loss': 1},
                       'inv_r': 0.01,
                       'max_slope': 0.03,
                       'others_slope': 0.01,
                       'rounding_slope':0.001,
                       'adj_mask_offset': None,
                       'l_loss': 0.01,
                       'l_const': 100,
                       'l_prop': 100,
                       'stop_loss': 0.3,
                       'stop_prop': 100,
                       'adj_eps': 0.05,
                       'show_losses': False,
                       'max_attempts': 3,
                       'rounding': 'step',
                       'bond_multiplier': 0.3,
                       'output_multiplier': 1,
                       'true_loss': True,
                       'embed': True,
                       'strictly_limit_bonds': True}
