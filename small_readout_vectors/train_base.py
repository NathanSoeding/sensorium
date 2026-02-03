import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from nnfabrik.builder import get_data, get_trainer

from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer
from models.bottleneck import Bottleneck

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--output_dir', type=str, default="runs/exp1")
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--feature_reg_weight', type=float, default=3.0)
    parser.add_argument('--gamma_sigma', type=float, default=0.25)
    parser.add_argument('--topo_w', type=float, default=None)
    parser.add_argument('--final_nonlin', action='store_false', default=True)
    parser.add_argument('--use_wandb', action='store_false', default=True)
    parser.add_argument('--topo_k', type=int, default=None)
    parser.add_argument('--init_path', type=str, default=None)
    parser.add_argument('--bottleneck_layers', type=int, nargs='+', default=None)
    parser.add_argument('--barlow_w', type=float, default=None)
    parser.add_argument('--finetune_lr_scale', type=float, default=None)
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

    model_config = {
        'hidden_channels': args.hidden_channels, # original sensorium was 64ch
        'depth_separable': False,
        'use_avg_reg': False,
        'laplace_padding': None,
        'momentum': 0.75,
        'final_nonlinearity': args.final_nonlin,
        'nonlinearity_type': 'AdaptiveELU',
        'pad_input': False,
        'stack': -1,
        'layers': 4,
        'input_kern': 11, #  original sensorium was 'input_kern': 9,
        'gamma_input': 6.3831, # this should not influence the performance based on previous experience
        
        'feature_reg_weight': args.feature_reg_weight, # this is the one I actually would like to tune!
        
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
        'gamma_sigma' : args.gamma_sigma,
    }

    if args.bottleneck_layers is not None:
        in_dim = model_config['hidden_channels']
        hidden_dims = args.bottleneck_layers
        model_config['bottleneck'] = Bottleneck(
            in_dim=in_dim, 
            hidden_dims=hidden_dims[:-1], 
            embedding_dim=hidden_dims[-1], 
        )
        wandb_name = f'bottleneck{hidden_dims[-1]}'

    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
        
    if args.init_path is not None:    
        model.load_state_dict(torch.load(args.init_path))

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_name': wandb_name or 'base model',
        'topographic_loss_w': args.topo_w, 
        'topographic_loss_k': args.topo_k, 
        'barlow_loss_w': args.barlow_w, 
        'use_wandb': args.use_wandb, 
    }
    trainer_config['wandb_config'] = model_config | trainer_config

    if args.finetune_lr_scale is not None:
        autoenc_params = []
        finetune_params = []

        for name, param in model.named_parameters():
            if 'autoencoder' in name:
                autoenc_params.append(param)
            else:
                finetune_params.append(param)

        base_lr = trainer_config['lr_init']
        trainer_config['optimizer'] = torch.optim.Adam([
            {'params': autoenc_params, 'lr': base_lr}, 
            {'params': finetune_params, 'lr': base_lr * args.finetune_lr_scale},
        ])

    #validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )

    torch.save(model.state_dict(), f'{args.output_dir}/model_weights.pth')

if __name__ == "__main__":
    main()