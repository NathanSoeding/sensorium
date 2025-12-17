import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from nnfabrik.builder import get_data, get_trainer

from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer

from autoencoder import Autoenc

def get_args():
    parser = argparse.ArgumentParser(description="Pytorch Training Script")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--output_dir', type=str, default="runs/exp1")
    parser.add_argument('--base_path', type=str, default='runs/base/exp0/model_weights.pth')
    
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64])
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--nonlinearity', type=str, default='ReLU')
    return parser.parse_args()

def main():
    args = get_args()
    print(args)

    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    device = args.device
    torch.cuda.set_device(device)

    basepath = "/srv/user/polina/sensorium/sensorium/notebooks/data/"

    # as filenames, we'll select all 7 datasets
    filenames = [
        os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file
    ]

    dataset_fn = "sensorium.datasets.static_loaders"
    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": True,
        "include_eye_position": True,
        "batch_size": 128,
        "scale": 0.25, 
    }
    dataloaders = get_data(dataset_fn, dataset_config)
    data_keys = list(dataloaders['train'].keys())

    model_config = {
        'hidden_channels': 128, # original sensorium was 64ch
        'depth_separable': False,
        'use_avg_reg': False,
        'laplace_padding': None,
        'momentum': 0.75,
        'final_nonlinearity': True,
        'nonlinearity_type': 'AdaptiveELU',
        'pad_input': False,
        'stack': -1,
        'layers': 4,
        'input_kern': 11, #  original sensorium was 'input_kern': 9,
        'gamma_input': 6.3831, # this should not influence the performance based on previous experience
        
        'feature_reg_weight': 3, # this is the one I actually would like to tune!
        
        'hidden_kern': 7,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True
        },
        
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'shifter': True,
        'batch_norm_scale': [True, True, True, False],
        'core_bias': [True, True, True, False],
        'regularizer_type': "adaptive_log_norm",
        'gamma_sigma' : 0.25,
    }

    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
    model.load_state_dict(torch.load(args.base_path))

    # Freeze entire model
    for param in model.parameters():
        param.requires_grad = False
    
    autoencoder_config = {
        'latent_dim': args.latent_dim, 
        'hidden_dims': args.hidden_dims, 
        'batch_norm': args.batch_norm,
        'nonlinearity': args.nonlinearity,
    }

    autoencoder = Autoenc(input_dim=model_config['hidden_channels'], **autoencoder_config)

    for key in data_keys:
        model.readout[key].autoencoder = autoencoder
    
    # Neuron idcs train/test splits
    num_val_neurons = 500
    train_neurons = {}
    validation_neurons = {}
    data_keys = list(dataloaders['train'].keys())

    for key in data_keys:
        responses = next(iter(dataloaders['train'][key]))[1]
        num_neurons = responses.shape[1]
        #print(num_neurons)
        
        idcs = np.random.permutation(num_neurons)
        train_neurons[key] = idcs[num_val_neurons:]
        validation_neurons[key] = idcs[:num_val_neurons]

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_config': autoencoder_config,
        'wandb_name': f'autoencoder{autoencoder_config['latent_dim']}',
        'train_neurons': train_neurons, 
        'validation_neurons': validation_neurons, 
    }

    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )
    torch.save(autoencoder.state_dict(), f'{args.output_dir}/autoencoder_weights.pth')
    with open(f'{args.output_dir}/data.yaml', 'w') as f:
        data = autoencoder_config | {'validation_score': validation_score}
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    main()