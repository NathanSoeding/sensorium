import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import wandb
from nnfabrik.builder import get_data, get_trainer

from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer
from models.bottleneck import Bottleneck

from models.whitener import Whitener

def train(
    seed=42,
    device='cuda:7',
    output_dir='runs/exp1',
    hidden_channels=128,
    feature_reg_weight=3.0,
    gamma_sigma=0.25,
    topo_w=None,
    final_nonlin=True,
    no_wandb=False,
    wandb_run_name='run',
    topo_k=None,
    init_path=None,
    only_load=None,
    bottleneck_layers=None,
    barlow_w=None,
    finetune_lr_scale=None,
    padding=None,
    batch_size=128,
    two_mlps=False,
    identity=False,
    exclude_behaviour=False,
    sample_first=False,
    whitener=False,
    whitener_ema_decay=None,
    reg_type='adaptive_log_norm',
    lp_p=0.5,
    lp_eps=1e-3,
    reg_start=None,
    reg_end=None,
    freeze_core=False,
    freeze_shifter=False,
    readout_type='gaussian',
    optimizer=None,
    adamw_reg=1e-2,
    spatial_reg_weight=0.0,
    temperature=1.0,
    factorize_spatial=False,
    shift_noise_scale=None,
    retinotopy_type=None,
    retinotopy_d=30,
    retinotopy_layers=1,
    max_freq=None,
    input_kern=11,
    hidden_kern=7,
):
    random_seed = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = device
    torch.cuda.set_device(device)

    use_wandb = not no_wandb

    basepath = "/srv/user/polina/sensorium/sensorium/notebooks/data/"

    # as filenames, we'll select all 7 datasets
    filenames = [
        os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file
    ]

    dataset_fn = "sensorium.datasets.static_loaders"
    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": not exclude_behaviour,
        "include_eye_position": True,
        "batch_size": batch_size,
        "scale": 0.25,
    }

    dataloaders = get_data(dataset_fn, dataset_config)

    model_config = {
        'hidden_channels': hidden_channels, # original sensorium was 64ch
        'depth_separable': False,
        'use_avg_reg': False,
        'laplace_padding': None,
        'momentum': 0.75,
        'final_nonlinearity': final_nonlin,
        'nonlinearity_type': 'AdaptiveELU',
        'pad_input': False,
        'hidden_padding': padding, 
        'stack': -1,
        'layers': 4,
        'input_kern': input_kern, #  original sensorium was 'input_kern': 9,
        'gamma_input': 6.3831, # this should not influence the performance based on previous experience
        
        'feature_reg_weight': feature_reg_weight, # this is the one I actually would like to tune!
        
        'hidden_kern': hidden_kern,
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
        'regularizer_type': reg_type,
        'gamma_sigma': gamma_sigma,
        'lp_p': lp_p,
        'lp_eps': lp_eps,
        'readout_type': readout_type,
        'spatial_reg_weight': spatial_reg_weight,
        'temperature': temperature,
        'factorize_spatial': factorize_spatial,
        'shift_noise_scale': shift_noise_scale,
    }

    if retinotopy_type is not None:
        model_config['retinotopy_spatial'] = {
            'type': retinotopy_type,
            'max_freq': max_freq,
            'hidden_features': retinotopy_d,
            'hidden_layers': retinotopy_layers,
        }

    if whitener:
        model_config['whitener'] = Whitener(model_config['hidden_channels'], whitener_ema_decay)
        #model_config['batch_norm_scale'] = True
        #model_config['core_bias'] = True

    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
    print(model)
        
    if init_path is not None:
        if only_load is None:
            print(f'loading {init_path}')
            model.load_state_dict(torch.load(init_path, map_location=device))
        else:
            print(f'loading {only_load} from {init_path}')
            ckpt = torch.load(init_path, map_location=device)
            model_dict = model.state_dict()
            
            # Keep only matching keys (and optionally matching shapes)
            filtered_ckpt = {
                k: v
                for k, v in ckpt.items()
                if any(s in k for s in only_load)
            }
            model_dict.update(filtered_ckpt)
            model.load_state_dict(model_dict)

    if bottleneck_layers is not None:
        print(f'creating bottleneck with dims {bottleneck_layers}')
        in_dim = model_config['hidden_channels']
        hidden_dims = bottleneck_layers
        bottleneck = Bottleneck(
            in_dim=in_dim, 
            hidden_dims=hidden_dims[:-1], 
            embedding_dim=hidden_dims[-1], 
            weight_sharing=not two_mlps, 
            identity=identity,
            sample_first=sample_first,
        )
        print(bottleneck)
        for key in dataloaders['train'].keys():
            model.readout[key].bottleneck = bottleneck

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_name': wandb_run_name,
        'topographic_loss_w': topo_w, 
        'topographic_loss_k': topo_k, 
        'barlow_loss_w': barlow_w, 
        'use_wandb': use_wandb, 
        'regularizer_warmup_start': reg_start,
        'regularizer_warmup_end': reg_end,
        'optimizer': optimizer,
        'adamw_reg': adamw_reg,
    }
    trainer_config['wandb_config'] = model_config | trainer_config

    if finetune_lr_scale is not None:
        if finetune_lr_scale == 0:
            # finetune_lr = 0 -> Freeze params
            print('freezing everything excect bottleneck')
            for name, param in model.named_parameters():
                if 'bottleneck' not in name:
                    param.requires_grad = False
            
        else:
            # finetune_lr > 0 -> split params
            print('split params')
            bottleneck_params = []
            finetune_params = []

            for name, param in model.named_parameters():
                if 'bottleneck' in name:
                    bottleneck_params.append(param)
                else:
                    finetune_params.append(param)

            base_lr = trainer_config['lr_init']
            trainer_config['optimizer'] = torch.optim.Adam([
                {'params': bottleneck_params, 'lr': base_lr}, 
                {'params': finetune_params, 'lr': base_lr * finetune_lr_scale},
            ])

    if freeze_core:
        print('freezing core')
        for name, param in model.named_parameters():
            if 'core' in name:
                print(f'freezing {name}')
                param.requires_grad = False
    if freeze_shifter:
        print('freezing shifter')
        for name, param in model.named_parameters():
            if 'shifter' in name:
                print(f'freezing {name}')
                param.requires_grad = False

    #validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )

    torch.save(model.state_dict(), f'{output_dir}/model_weights.pth')
    return validation_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--input_kern', type=int)
    parser.add_argument('--hidden_kern', type=int)
    parser.add_argument('--feature_reg_weight', type=float)
    parser.add_argument('--retinotopy_d', type=int)
    parser.add_argument('--retinotopy_layers', type=int)
    parser.add_argument('--max_freq', type=int)
    args = parser.parse_args()

    wandb.init(project='small readout vectors', name='sweep')
    config = wandb.config

    out_dir = os.path.join("runs/sweeps/factorized_sweep", wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)
    score = train(
        seed=0,
        device='cuda:0',
        output_dir=out_dir,
        wandb_run_name=wandb.run.id,
        hidden_channels=args.hidden_channels,
        input_kern=args.input_kern,
        hidden_kern=args.hidden_kern,
        feature_reg_weight=args.feature_reg_weight,
        readout_type='factorized',
        retinotopy_type='fourier',
        retinotopy_d=args.retinotopy_d,
        retinotopy_layers=args.retinotopy_layers,
        max_freq=args.max_freq,
        whitener=True,
        whitener_ema_decay=0.003,
        no_wandb=True,
    )
    wandb.log({'score': score})

if __name__ == "__main__":
    main()