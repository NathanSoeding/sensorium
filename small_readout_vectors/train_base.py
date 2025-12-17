import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from nnfabrik.builder import get_data, get_trainer

from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument("--output_dir", type=str, default="runs/exp1")
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

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_config': model_config,
        'wandb_name': 'base model',
    }

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