import os
import torch
import argparse

from sensorium.utility.utils import get_data, set_random_seed
from sensorium.models.models import stacked_core_full_gauss_readout
from sensorium.training.trainers import standard_trainer

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
    parser.add_argument('--whitener', action='store_true', default=False)
    parser.add_argument('--whitener_momentum', type=float, default=0.003)
    parser.add_argument('--readout_type', type=str, default='gaussian')
    parser.add_argument('--more_data', action='store_true', default=False)
    parser.add_argument('--shifter_bias', action='store_true', default=False)
    parser.add_argument('--shifter_features', type=int, default=5)
    parser.add_argument('--shifter_layers', type=int, default=1)
    parser.add_argument('--init_gain', type=float, default=0.1)
    
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
        os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file
    ]

    if args.more_data:
        more_basepath = "/user/turishcheva/more_data_like_sensorium_2022"
        for file in [
            "static20457-5-9-94c6ff995dac583098847cfecd43e7b6",
            "static20622-2-14-94c6ff995dac583098847cfecd43e7b6",
            "static20892-10-10-94c6ff995dac583098847cfecd43e7b6",
            "static22223-2-15-94c6ff995dac583098847cfecd43e7b6",
            "static22564-2-12-94c6ff995dac583098847cfecd43e7b6",
            "static22620-4-15-94c6ff995dac583098847cfecd43e7b6",
            "static23555-5-12-94c6ff995dac583098847cfecd43e7b6",
        ]:
            filenames.append(os.path.join(more_basepath, file))

    dataset_fn = "sensorium.datasets.static_loaders"
    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": True,
        "include_eye_position": True,
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
        'shifter_bias': args.shifter_bias,
        'hidden_channels_shifter': args.shifter_features,
        'shift_layers': args.shifter_layers,
        'batch_norm_scale': [True, True, True, False],
        'core_bias': [True, True, True, False],
        'regularizer_type': 'adaptive_log_norm',
        'gamma_sigma': args.gamma_sigma,
        #'readout_type': args.readout_type,
        'hidden_channels_shifter': args.shifter_features,
        'shift_layers': args.shifter_layers,
        # 'discretized_spatial': args.discretized,
        # 'kernel_size': args.kernel_size,
        # 'sigma': args.sigma,
        'init_gain': args.init_gain,
        'whitener': args.whitener,
        'whitener_momentum': args.whitener_momentum,
    }
    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
    print(model)

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
    }
    trainer_config['wandb_config'] = model_config | trainer_config

    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )
    torch.save(model.state_dict(), f'{args.output_dir}/model_weights.pth')

if __name__ == "__main__":
    main()
    