import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from nnfabrik.builder import get_data, get_trainer

from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer
from models.bottleneck import Bottleneck

from models.whitener import Whitener

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--output_dir', type=str, default="runs/exp1")
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--feature_reg_weight', type=float, default=3.0)
    parser.add_argument('--gamma_sigma', type=float, default=0.25)
    parser.add_argument('--topo_w', type=float, default=None)
    parser.add_argument('--final_nonlin', action='store_false', default=True)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_run_name', type=str, default='run')
    parser.add_argument('--topo_k', type=int, default=None)
    parser.add_argument('--init_path', type=str, default=None)
    parser.add_argument('--only_load', type=str, nargs='+', default=None)
    parser.add_argument('--bottleneck_layers', type=int, nargs='+', default=None)
    parser.add_argument('--barlow_w', type=float, default=None)
    parser.add_argument('--finetune_lr_scale', type=float, default=None)
    parser.add_argument('--padding', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--two_mlps', action='store_true', default=False)
    parser.add_argument('--identity', action='store_true', default=False)
    parser.add_argument('--exclude_behaviour', action='store_true', default=False)
    parser.add_argument('--sample_first', action='store_true', default=False)
    parser.add_argument('--whitener', action='store_true', default=False)
    parser.add_argument('--whitener_ema_decay', type=float, default=0.003)
    parser.add_argument('--reg_type', type=str, default="adaptive_log_norm")
    parser.add_argument('--lp_p', type=float, default=0.5)
    parser.add_argument('--lp_eps', type=float, default=1e-3)
    parser.add_argument('--reg_start', type=int, default=None)
    parser.add_argument('--reg_end', type=int, default=None)
    parser.add_argument('--freeze_core', action='store_true', default=False)
    parser.add_argument('--freeze_shifter', action='store_true', default=False)
    parser.add_argument('--freeze_spatial', action='store_true', default=False)
    parser.add_argument('--readout_type', type=str, default='gaussian')
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--adamw_reg', type=float, default=1e-2)
    parser.add_argument('--spatial_reg_weight', type=float, default=0.0)
    parser.add_argument('--factorize_spatial', action='store_true', default=False)
    parser.add_argument('--shift_noise_scale', type=float, default=None)
    parser.add_argument('--retinotopy', action='store_true', default=False)
    parser.add_argument('--retinotopy_d', type=int, default=30)
    parser.add_argument('--retinotopy_layers', type=int, default=0)
    parser.add_argument('--max_freq', type=int, default=4)
    parser.add_argument('--input_kern', type=int, default=11)
    parser.add_argument('--fourier', action='store_true', default=False)
    parser.add_argument('--hard', action='store_true', default=False)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--min_temp', type=float, default=None)
    parser.add_argument('--temp_decay_t', type=int, default=None)
    parser.add_argument('--normalize_logits', action='store_true', default=False)
    parser.add_argument('--retinotopy_shifter', action='store_true', default=False)
    parser.add_argument('--perspective', action='store_true', default=False)
    parser.add_argument('--retina_degree', type=int, default=75)
    parser.add_argument('--retina_mlp_features', type=int, default=16)
    parser.add_argument('--retina_mlp_layers', type=int, default=3)
    parser.add_argument('--entropy_reg', action='store_true', default=False)
    parser.add_argument('--entropy_reg_weight', type=float, default=1.0)
    parser.add_argument('--no_shifter', action='store_true', default=False)
    parser.add_argument('--no_shift_bias', action='store_true', default=False)
    parser.add_argument('--shifter_features', type=int, default=5)
    parser.add_argument('--shifter_layers', type=int, default=3)
    parser.add_argument('--gamma_shifter', type=float, default=0.0)
    parser.add_argument('--no_diag', action='store_true', default=False)
    parser.add_argument('--com_reg_weight', type=float, default=0.0)
    parser.add_argument('--gaussian', action='store_true', default=False)
    parser.add_argument('--predict_sigma', action='store_true', default=False)
    parser.add_argument('--init_sigma', type=float, default=1.0)
    parser.add_argument('--cp_every_epoch', action='store_true', default=False)
    parser.add_argument('--cp_path', type=str, default=None)
    parser.add_argument('--discretized', action='store_true', default=False)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--init_gain', type=float, default=1.0)
    parser.add_argument('--more_data', action='store_true', default=False)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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
        "include_behavior": not args.exclude_behaviour,
        "include_eye_position": not args.no_shifter,
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
        'final_nonlinearity': args.final_nonlin,
        'nonlinearity_type': 'AdaptiveELU',
        'pad_input': False,
        'hidden_padding': args.padding, 
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
        'shifter': not (args.retinotopy_shifter or args.perspective or args.no_shifter),
        'batch_norm_scale': [True, True, True, False],
        'core_bias': [True, True, True, False],
        'regularizer_type': args.reg_type,
        'gamma_sigma': args.gamma_sigma,
        'lp_p': args.lp_p,
        'lp_eps': args.lp_eps,
        'readout_type': args.readout_type,
        'spatial_reg_weight': args.spatial_reg_weight,
        'factorize_spatial': args.factorize_spatial,
        'shift_noise_scale': args.shift_noise_scale,
        'fourier_spatial': args.fourier,
        'fourier_max_freq': args.max_freq,
        'init_temp': args.temp,
        'hard': args.hard,
        'normalize_logits': args.normalize_logits,
        'perspective': args.perspective,
        'retina_degree': args.retina_degree,
        'retina_mlp_features': args.retina_mlp_features,
        'retina_mlp_layers': args.retina_mlp_layers,
        'entropy_reg': args.entropy_reg,
        'entropy_reg_weight': args.entropy_reg_weight,
        'shifter_bias': not args.no_shift_bias,
        'hidden_channels_shifter': args.shifter_features,
        'shift_layers': args.shifter_layers,
        'gamma_shifter': args.gamma_shifter,
        'diagonal': not args.no_diag,
        'com_reg_weight': args.com_reg_weight,
        'gaussian_spatial': args.gaussian,
        'predict_sigma': args.predict_sigma,
        'retinotopy_init_sigma': args.init_sigma,
        'discretized_spatial': args.discretized,
        'kernel_size': args.kernel_size,
        'sigma': args.sigma,
        'init_gain': args.init_gain,
    }

    if args.retinotopy:
        model_config['retinotopy_spatial'] = {
            'hidden_features': args.retinotopy_d,
            'hidden_layers': args.retinotopy_layers,
        }
    if args.retinotopy_shifter:
        model_config['retinotopy_spatial']['in_dim'] = 4
    else:
        model_config['retinotopy_spatial']['in_dim'] = None

    if args.whitener:
        model_config['whitener'] = Whitener(model_config['hidden_channels'], args.whitener_ema_decay)
        #model_config['batch_norm_scale'] = True
        #model_config['core_bias'] = True

    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
    print(model)
        
    if args.init_path is not None:
        if args.only_load is None:
            print(f'loading {args.init_path}')
            model.load_state_dict(torch.load(args.init_path, map_location=device))
        else:
            print(f'loading {args.only_load} from {args.init_path}')
            ckpt = torch.load(args.init_path, map_location=device)
            model_dict = model.state_dict()
            
            # Keep only matching keys (and optionally matching shapes)
            filtered_ckpt = {
                k: v
                for k, v in ckpt.items()
                if any(s in k for s in args.only_load)
            }
            model_dict.update(filtered_ckpt)
            model.load_state_dict(model_dict)

    if args.bottleneck_layers is not None:
        print(f'creating bottleneck with dims {args.bottleneck_layers}')
        in_dim = model_config['hidden_channels']
        hidden_dims = args.bottleneck_layers
        bottleneck = Bottleneck(
            in_dim=in_dim, 
            hidden_dims=hidden_dims[:-1], 
            embedding_dim=hidden_dims[-1], 
            weight_sharing=not args.two_mlps, 
            identity=args.identity,
            sample_first=args.sample_first,
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
        'wandb_name': args.wandb_run_name,
        'topographic_loss_w': args.topo_w, 
        'topographic_loss_k': args.topo_k, 
        'barlow_loss_w': args.barlow_w, 
        'use_wandb': use_wandb, 
        'regularizer_warmup_start': args.reg_start,
        'regularizer_warmup_end': args.reg_end,
        'optimizer': args.optimizer,
        'adamw_reg': args.adamw_reg,
        'init_temp': args.temp, 
        'min_temp': args.min_temp,
        'temp_decay_t': args.temp_decay_t,
        'cp_every_epoch': args.cp_every_epoch,
        'cp_path': args.cp_path,
    }
    if args.cp_every_epoch:
        os.makedirs(args.cp_path, exist_ok=True)

    trainer_config['wandb_config'] = model_config | trainer_config

    if args.finetune_lr_scale is not None:
        if args.finetune_lr_scale == 0:
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
                {'params': finetune_params, 'lr': base_lr * args.finetune_lr_scale},
            ])

    if args.freeze_core:
        print('freezing core')
        for name, param in model.named_parameters():
            if 'core' in name:
                print(f'freezing {name}')
                param.requires_grad = False
    if args.freeze_shifter:
        print('freezing shifter')
        for name, param in model.named_parameters():
            if 'shifter' in name:
                print(f'freezing {name}')
                param.requires_grad = False
    if args.freeze_spatial:
        print('freezing spatial')
        for name, param in model.named_parameters():
            if 'spatial' in name:
                print(f'freezing {name}')
                param.requires_grad = False

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