import os
import json
import torch
import argparse

from sensorium.utility import get_data, set_random_seed, build_dedup_tensors
from sensorium.models import stacked_core_full_gauss_readout, stacked_core_factorized_readout
from sensorium.models.zig_model import stacked_core_zig_gauss_readout, stacked_core_zig_factorized_readout
from sensorium.training import standard_trainer
import pickle

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default="runs/")
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--input_kern', type=int, default=9)
    parser.add_argument('--feature_reg_weight', type=float, default=1.0)
    parser.add_argument('--gamma_sigma', type=float, default=0.25)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_run_name', type=str, default='run')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--disable_whitener', action='store_true', default=False)
    parser.add_argument('--whitener_momentum', type=float, default=0.003)

    parser.add_argument('--readout_type', type=str, default='factorized')

    parser.add_argument('--retinotopy_features', type=int, default=30)
    parser.add_argument('--retinotopy_layers', type=int, default=1)

    parser.add_argument('--temp_per_neuron', action='store_true', default=False)
    parser.add_argument('--readout_kernel_size', type=int, default=7)
    parser.add_argument('--readout_kernel_sigma', type=float, default=2.0)
    parser.add_argument('--smoothness_reg_weight', type=float, default=0.0)

    parser.add_argument('--more_data', action='store_true', default=False)
    parser.add_argument('--shifter_bias', action='store_true', default=False)
    parser.add_argument('--shifter_features', type=int, default=5)
    parser.add_argument('--shifter_layers', type=int, default=1)
    parser.add_argument('--init_gain', type=float, default=0.1)

    parser.add_argument('--include_kldivergence', action='store_true', default=False)
    parser.add_argument('--cluster_number', type=int, default=10)
    parser.add_argument('--dec_starting_epoch', type=int, default=10)
    parser.add_argument('--base_multiplier', type=float, default=4e3)
    parser.add_argument('--dedup_mode', type=str, choices=['none', 'mean', 'random_representative'], default='none')

    parser.add_argument('--use_zig_loss', action='store_true', default=False)
    parser.add_argument('--gamma_params_dir', type=str, default='sensorium/data/gamma_params')

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
    
    if args.more_data:
        filenames = [
            os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file and '26872-17-20' not in file
        ]
    else:
        filenames = [
            os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file
        ]

    if args.more_data:
        more_basepath = "/user/turishcheva/more_data_like_sensorium_2022"
        # we should exclude mouse 20892 since it has not only V1 but also other areas!
        # We also excluded mouse 20622 since it has data from L4 and not L2/3 as all the other mice
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

    print(len(filenames), filenames)

    dataset_fn = "sensorium.datasets.static_loaders"
    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": True,
        "include_eye_position": True,
        "exclude_eye_position_paths": [
            '/srv/user/polina/sensorium/sensorium/notebooks/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip'
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
        'whitener': not args.disable_whitener,
        'whitener_momentum': args.whitener_momentum,
    }

    if args.readout_type == 'gaussian':
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
        model_config['temp_per_neuron'] = args.temp_per_neuron
        model_config['readout_kernel_size'] = args.readout_kernel_size
        model_config['readout_kernel_sigma'] = args.readout_kernel_sigma
        model_config['smoothness_reg_weight'] = args.smoothness_reg_weight
        model = stacked_core_factorized_readout(dataloaders, random_seed, **model_config)

    if args.readout_type == 'zig':
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
        model_config['gamma_params_dir'] = args.gamma_params_dir
        model = stacked_core_zig_gauss_readout(dataloaders, random_seed, **model_config)

    if args.readout_type == 'zig_factorized':
        model_config['temp_per_neuron'] = args.temp_per_neuron
        model_config['readout_kernel_size'] = args.readout_kernel_size
        model_config['readout_kernel_sigma'] = args.readout_kernel_sigma
        model_config['smoothness_reg_weight'] = args.smoothness_reg_weight
        model_config['gamma_params_dir'] = args.gamma_params_dir
        model = stacked_core_zig_factorized_readout(dataloaders, random_seed, **model_config)
    print(model)

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'log_smoothness': args.smoothness_reg_weight > 0.0,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_name': args.wandb_run_name,
        'use_wandb': use_wandb, 

        'include_kldivergence': args.include_kldivergence,
        'cluster_number': args.cluster_number,
        'dec_starting_epoch': args.dec_starting_epoch,
        'base_multiplier': args.base_multiplier,

        'use_zig_loss': args.use_zig_loss,
    }
    trainer_config['wandb_config'] = model_config | trainer_config

    if args.include_kldivergence and args.dedup_mode != 'none':
        # kept out of wandb_config above: dedup_info is a large per-session tensor dict, not
        # small serializable run metadata
        trainer_config['dedup_mode'] = args.dedup_mode
        trainer_config['dedup_info'] = build_dedup_tensors(list(dataloaders['train'].keys()), device)
    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )
    torch.save(model.state_dict(), f'{args.output_dir}/model_weights.pth')
    with open(f'{args.output_dir}/output_dict.pkl', 'wb') as f:
        pickle.dump(trainer_output, f)

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
    