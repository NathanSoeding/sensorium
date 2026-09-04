import os
import json
import torch
import argparse

from sensorium.utility import get_data, set_random_seed
from sensorium.models import stacked_core_full_gauss_readout, stacked_core_factorized_readout
from sensorium.training import standard_trainer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default="runs/")
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--input_kern', type=int, default=9)
    parser.add_argument('--feature_reg_weight', type=float, default=3.0)
    parser.add_argument('--gamma_sigma', type=float, default=0.25)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_run_name', type=str, default='run')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--whitener', action='store_true', default=False)
    parser.add_argument('--whitener_momentum', type=float, default=0.003)
    parser.add_argument('--cp_every_epoch', action='store_true', default=False)
    parser.add_argument('--no_retinotopy', action='store_true', default=False)

    parser.add_argument('--readout_type', type=str, default='factorized')

    parser.add_argument('--retinotopy_features', type=int, default=30)
    parser.add_argument('--retinotopy_layers', type=int, default=1)

    parser.add_argument('--spatial_init_noise', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temp_per_neuron', action='store_true', default=False)
    parser.add_argument('--readout_kernel_size', type=int, default=17)
    parser.add_argument('--readout_kernel_sigma', type=float, default=4.0)
    parser.add_argument('--smoothness_reg_weight', type=float, default=0.0)
    parser.add_argument('--entropy_reg_weight', type=float, default=0.0)

    parser.add_argument('--more_data', action='store_true', default=False)
    parser.add_argument('--shifter_bias', action='store_true', default=False)
    parser.add_argument('--shifter_features', type=int, default=5)
    parser.add_argument('--shifter_layers', type=int, default=1)
    parser.add_argument('--init_gain', type=float, default=0.1)

    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--load_parts', type=str, nargs='+', default=None)
    parser.add_argument('--freeze_parts', type=str, nargs='+', default=None)
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    random_seed = args.seed
    set_random_seed(random_seed, deterministic=False)

    device = args.device
    torch.cuda.set_device(device)

    use_wandb = not args.no_wandb

    basepath = "/srv/user/polina/sensorium/sensorium/notebooks/data/"
    # as filenames, we'll select all 7 datasets
    filenames = [
        os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file # and (not args.more_data and ())
    ]
    filenames = [] 
    for file in [
        'static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        'static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
        'static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        'static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        'static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        'static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6', 
        'static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6',
    ]:
        if not (args.more_data and '26872' in file):  # if args.more_data exclude the no behav animal
            filenames.append(os.path.join(basepath, file))

    if args.more_data:
        more_basepath = "/user/turishcheva/more_data_like_sensorium_2022"
        for file in [
            "static20457-5-9-94c6ff995dac583098847cfecd43e7b6",
            # "static20622-2-14-94c6ff995dac583098847cfecd43e7b6",
            # "static20892-10-10-94c6ff995dac583098847cfecd43e7b6",
            "static22223-2-15-94c6ff995dac583098847cfecd43e7b6",
            "static22564-2-12-94c6ff995dac583098847cfecd43e7b6",
            "static22620-4-15-94c6ff995dac583098847cfecd43e7b6",
            "static23555-5-12-94c6ff995dac583098847cfecd43e7b6",
        ]:
            filenames.append(os.path.join(more_basepath, file))
    for file in filenames:
        print(file)

    dataset_fn = "sensorium.datasets.static_loaders"
    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": True,
        "include_eye_position": True,
        "exclude_eye_position_paths": [
            '/srv/user/polina/sensorium/sensorium/notebooks/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6'
        ],
        "batch_size": args.batch_size,
        "scale": 0.25,
    }
    dataloaders = get_data(dataset_fn, dataset_config)
    
    model_config = {
        'hidden_channels': args.hidden_channels, # original sensorium was 64ch
        'depth_separable': False,
        'use_avg_reg': False,
        'laplace_padding': None,
        'momentum': 0.75,
        'final_nonlinearity': True,
        'nonlinearity_type': 'AdaptiveELU',
        'pad_input': False,
        'hidden_padding': None, 
        'stack': -1,
        'layers': 4,
        'input_kern': args.input_kern, #  original sensorium was 'input_kern': 9,
        'gamma_input': 6.3831, # this should not influence the performance based on previous experience
        
        'feature_reg_weight': args.feature_reg_weight, # this is the one I actually would like to tune!
        
        'hidden_kern': 7,
        
        'shifter': True,
        'shifter_bias': args.shifter_bias,
        'hidden_channels_shifter': args.shifter_features,
        'shift_layers': args.shifter_layers,
        'batch_norm_scale': [True, True, True, False],
        'core_bias': [True, True, True, False],
        'regularizer_type': 'adaptive_log_norm',
        'gamma_sigma': args.gamma_sigma,
        #'readout_type': args.readout_type,
        # 'discretized_spatial': args.discretized,
        # 'kernel_size': args.kernel_size,
        # 'sigma': args.sigma,
        'init_gain': args.init_gain,
        'whitener': args.whitener,
        'whitener_momentum': args.whitener_momentum,
    }

    if args.readout_type == 'gaussian':
        if not args.no_retinotopy:
            model_config['grid_mean_predictor'] = {
                'type': 'cortex',
                'input_dimensions': 2,
                'hidden_layers': args.retinotopy_layers,
                'hidden_features': args.retinotopy_features,
                'final_tanh': True
            }
        model_config['init_sigma'] = 0.1
        model_config['init_mu_range'] = 0.3
        model_config['gauss_type'] = 'full'
        model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
    
    if args.readout_type == 'factorized':
        model_config['spatial_init_noise'] = args.spatial_init_noise
        model_config['temperature'] = args.temperature
        model_config['temp_per_neuron'] = args.temp_per_neuron
        model_config['readout_kernel_size'] = args.readout_kernel_size
        model_config['readout_kernel_sigma'] = args.readout_kernel_sigma
        model_config['smoothness_reg_weight'] = args.smoothness_reg_weight
        model_config['entropy_reg_weight'] = args.entropy_reg_weight
        model_config['retinotopy'] = not args.no_retinotopy
        model = stacked_core_factorized_readout(dataloaders, random_seed, **model_config)
    print(model)

    if args.model_checkpoint is not None:
        state_dict = torch.load(args.model_checkpoint, map_location=device)
        
        if args.load_parts is not None:
            state_dict = {
                k: v for k, v in state_dict.items()
                if any([part in k for part in args.load_parts])
            }
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        for k in state_dict.keys():
            print(f"Loaded {k}")
        print(f"Missing keys (not found in checkpoint / not loaded): {missing}")
        print(f"Unexpected keys (in checkpoint but not in model): {unexpected}")

    if args.freeze_parts is not None:
        for name, param in model.named_parameters():
            if any([part in name for part in args.freeze_parts]):
                print(f'Freezing {name}')
                param.requires_grad = False

    checkpoint_dir = f'{args.output_dir}/cps'
    if args.cp_every_epoch:
        os.makedirs(checkpoint_dir, exist_ok=True)

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_name': args.wandb_run_name,
        'use_wandb': use_wandb, 
        'cp_every_epoch': args.cp_every_epoch,
        'checkpoint_dir': checkpoint_dir,
    }
    trainer_config['wandb_config'] = model_config | trainer_config

    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )
    torch.save(model.state_dict(), f'{args.output_dir}/model_weights.pth')

    # Save everything needed to reconstruct the model
    full_config = {
        'readout_type': args.readout_type,
        'model_config': model_config,
        'random_seed': random_seed,
    }
    with open(f'{args.output_dir}/model_config.json', 'w') as f:
        json.dump(full_config, f, indent=2)

if __name__ == "__main__":
    main()
    