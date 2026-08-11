import json
import torch

from sensorium.models import stacked_core_full_gauss_readout, stacked_core_factorized_readout

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def load_model_from_config(run_dir, dataloaders, device='cuda:0', strict=True):
    with open(f'{run_dir}/model_config.json', 'r') as f:
        full_config = json.load(f)

    readout_type = full_config['readout_type']
    model_config = full_config['model_config']
    seed = full_config['random_seed']

    if readout_type == 'gaussian':
        model = stacked_core_full_gauss_readout(dataloaders, seed, **model_config)
    elif readout_type == 'factorized':
        model = stacked_core_factorized_readout(dataloaders, seed, **model_config)
    else:
        raise ValueError(f"Unknown readout_type: {readout_type}")

    state_dict = torch.load(f'{run_dir}/model_weights.pth', map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model

def knn_consistency(knn1, knn2, ks):
    N, k_max = knn1.shape
    #idcs = torch.randperm(N)
    #knn2 = knn2[idcs]
    
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

def get_knn_curve(all_features, k_range=None):
    if k_range is None:
        k_range = np.unique(np.geomspace(1, 100, num=20, dtype=int))

    knns = []
    for features in all_features:
        nn = NearestNeighbors(n_neighbors=k_range[-1]).fit(features)
        knn = nn.kneighbors(return_distance=False)
        knns.append(knn)

    #for leniency in leniency_range:
    x_idcs, y_idcs = np.tril_indices(len(all_features), k=-1)
    constistencies = []
    for x, y in zip(x_idcs, y_idcs):
        constistencies.append(knn_consistency(knns[x], knns[y], ks=k_range))
    constistencies = np.column_stack(constistencies)

    mean_curve = constistencies.mean(axis=1)
    std_curve = constistencies.std(axis=1, ddof=1)

    return (mean_curve, std_curve, constistencies), k_range

def get_ari_curve(n_clusters_range, all_features):
    def cluster(points, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(points)
        return labels
    
    mean_curve = []
    std_curve = []
    for n in n_clusters_range:
        labels = [cluster(features, n) for features in all_features]

        x_idcs, y_idcs = np.tril_indices(len(all_features), k=-1)
        aris = np.array([
            adjusted_rand_score(labels[x], labels[y]) for x, y in zip(x_idcs, y_idcs)
        ])

        ari_mean = aris.mean()
        ari_std = aris.std(ddof=1)
        mean_curve.append(ari_mean)
        std_curve.append(ari_std)
    
    return mean_curve, std_curve