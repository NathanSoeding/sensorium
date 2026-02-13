import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import argparse
import wandb

from nnfabrik.builder import get_data
from modified_neuralpredictors.create_model import stacked_core_full_gauss_readout
from modified_neuralpredictors.wandb_trainer import standard_trainer
from models.bottleneck import Bottleneck

def train(
    seed: int = 42,
    device: str = 'cuda:7',
    output_dir: str = "runs/exp1",
    hidden_channels: int = 128,
    feature_reg_weight: float = 3.0,
    gamma_sigma: float = 0.25,
    topo_w: float = None,
    final_nonlin: bool = True,
    no_wandb: bool = False,
    topo_k: int = None,
    init_path: str = None,
    bottleneck_layers: list = None,
    barlow_w: float = None,
    finetune_lr_scale: float = None,
    padding: int = None,
    batch_size: int = 128,
    per_neuron: bool = False,
    single_mlp: bool = False,
    no_feature_norm: bool = False, 
):
    random_seed = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = device
    torch.cuda.set_device(device)

    wandb_name = 'base model'
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
        "include_behavior": True,
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
        'input_kern': 11, #  original sensorium was 'input_kern': 9,
        'gamma_input': 6.3831, # this should not influence the performance based on previous experience
        
        'feature_reg_weight': feature_reg_weight, # this is the one I actually would like to tune!
        
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
        'gamma_sigma': gamma_sigma,
    }

    model = stacked_core_full_gauss_readout(dataloaders, random_seed, **model_config)
        
    if init_path is not None:
        print(f'loading {init_path}')
        model.load_state_dict(torch.load(init_path))

    if bottleneck_layers is not None:
        print(f'creating bottleneck with dims {bottleneck_layers}')
        in_dim = model_config['hidden_channels']
        hidden_dims = bottleneck_layers
        bottleneck = Bottleneck(
            in_dim=in_dim, 
            hidden_dims=hidden_dims[:-1], 
            embedding_dim=hidden_dims[-1], 
            weight_sharing=single_mlp, 
            feature_norm=not no_feature_norm, 
        )
        for key in dataloaders['train'].keys():
            model.readout[key].bottleneck = bottleneck

        wandb_name = f'bottleneck{hidden_dims[-1]}'

    trainer_config = {
        'max_iter': 200,
        'verbose': False,
        'lr_decay_steps': 4,
        'avg_loss': False,
        'lr_init': 0.009,
        'device': device, 
        'wandb_project': 'small readout vectors',
        'wandb_name': wandb_name,
        'topographic_loss_w': topo_w, 
        'topographic_loss_k': topo_k, 
        'barlow_loss_w': barlow_w, 
        'use_wandb': use_wandb, 
        'per_neuron': per_neuron, 
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

    #validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
    validation_score, trainer_output, state_dict = standard_trainer(
        model, 
        dataloaders, 
        seed=random_seed, 
        **trainer_config
    )

    torch.save(model.state_dict(), f'{output_dir}/model_weights.pth')

def metric(embeds):
    def knn_consistency(knn1, knn2, ks):
        N, k_max = knn1.shape
        
        assert ks[-1] <= k_max 
        ranking = np.full((N, N), N, dtype=np.int32)  # Shape (N, N)
        ranking[np.arange(N)[:, None], knn1] = np.arange(k_max)[None]
        overlaps = []

        for k in ks:
            neighbours = knn2[:, :k]  # Shape (N, k)
            shared = ranking[np.arange(N)[:, None], neighbours] < k  # Shape (N, k)
            num_shared = shared.sum() / N
            
            overlap = num_shared / k
            overlaps.append(overlap)

        return np.array(overlaps)

    def get_curve(k_range, all_features):
        knns = []
        for features in all_features:
            nn = NearestNeighbors(n_neighbors=k_range[-1]).fit(features)
            knn = nn.kneighbors(return_distance=False)
            knns.append(knn)

        #for leniency in leniency_range:
        x_idcs, y_idcs = np.tril_indices(len(all_features), k=-1)
        curve = np.zeros(len(k_range))
        for x, y in zip(x_idcs, y_idcs):
            curve += knn_consistency(knns[x], knns[y], ks=k_range) 
        
        curve /= len(x_idcs)

        return curve
    
    k_range = [100]
    curve = get_curve(k_range, embeds)
    return curve[0]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--output_root', type=str, default="runs/sweeps")
    args = parser.parse_args()

    wandb.init(project='small readout vectors', name='sweep')
    config = wandb.config

    bottleneck_layers = [config.bottleneck_hidden_dim, config.bottleneck_latent_dim] 

    run_base = os.path.join(args.output_root, wandb.run.id)

    seeds = [0, 42, 123]
    init_paths = [
        'runs/base/exp3/model_weights.pth', 
        'runs/base/exp4/model_weights.pth', 
        'runs/base/exp5/model_weights.pth', 
    ]
    model_paths = []
    for seed, init_path in zip(seeds, init_paths):
        run_output_dir = os.path.join(run_base, f"seed_{seed}")
        os.makedirs(run_output_dir, exist_ok=True)

        train(
            seed=seed, 
            device=args.device, 
            output_dir=run_output_dir, 
            hidden_channels=64, 
            init_path=init_path, 
            bottleneck_layers=bottleneck_layers,
            finetune_lr_scale=config.finetune_lr_scale, 
            batch_size=512, 
            single_mlp=config.weight_sharing, 
            feature_reg_weight=config.feature_reg_weight, 
            barlow_w=config.barlow_w, 
            no_wandb=True, 
        )
        model_paths.append(f'{run_output_dir}/model_weights.pth')

    bottleneck = Bottleneck(
        in_dim=64, 
        hidden_dims=bottleneck_layers[:-1], 
        embedding_dim=bottleneck_layers[-1], 
        weight_sharing=config.weight_sharing, 
    )
    
    def get_features(model_paths, combine_mice=True):
        all_features = []
        for model_path in model_paths:
            cp = torch.load(model_path)

            features = []
            for key, value in cp.items():
                if 'readout' in key and 'features' in key:
                    features.append(value.cpu().squeeze().T)
            if combine_mice:
                features = torch.cat(features)
            
            all_features.append(features)
        return all_features

    features = get_features(model_paths)
    embeds = []
    for features, path in zip(features, model_paths):
        bottleneck_params = {}
        for k, v in torch.load(path).items():
            if 'readout.23343-5-17.bottleneck' in k:
                new_k = k[30:]
                bottleneck_params[new_k] = v

        bottleneck.load_state_dict(bottleneck_params)
        bottleneck.eval()

        embeds.append(
            bottleneck.embed_neurons(
                features.unsqueeze(0).transpose(1, 2)
            ).squeeze().detach()
        )    
    
    score = metric(embeds)
    wandb.log({'knn_consistency': score})

if __name__ == "__main__":
    main()