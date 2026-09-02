import json
import torch

from sensorium.models import stacked_core_full_gauss_readout, stacked_core_factorized_readout

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def load_model_from_config(run_dir, dataloaders, device='cuda:0', strict=True, weights_path=None):
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

    if weights_path is None:
        state_dict = torch.load(f'{run_dir}/model_weights.pth', map_location=device)
    else:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model

def whiten(model, dataloaders, device, num_batches=1, t_readouts=False):
    # load batches and save feature vecs
    features = []
    data_keys = dataloaders.keys()
    for key in data_keys:
        batch_features = []
        for i, batch in enumerate(dataloaders[key]):
            if i == num_batches:
                break

            img, target, behav = batch[:3]
            img = img.to(device)
            
            if len(batch) == 4:
                pupil = batch[3]
                pupil = pupil.to(device)
            elif len(batch) == 3:
                pupil = None

            pred, feature = model(img, data_key=key, pupil_center=pupil, return_vec=True)
            feature = feature.detach().cpu()
            batch_features.append(feature)
        features.append(torch.cat(batch_features))

    X = torch.cat([f.flatten(0, 1) for f in features])
    if t_readouts:
        R = torch.cat([model.readout[k].features.squeeze().T.detach().cpu() for k in data_keys])
    else:
        R = torch.cat([model.readout[k].features.squeeze().detach().cpu() for k in data_keys])

    # whitening
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    
    eye = torch.eye(cov.shape[0])
    L = torch.linalg.cholesky(cov + 1e-5 * eye)
    Rw = R @ L
    return Rw

def knn_consistency(knn1, knn2, ks, chance_adjust=False):
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
        if chance_adjust:
            ev = k / N
            overlap = (overlap - ev) / (1 - ev)

        overlaps.append(overlap)

    return np.array(overlaps)

def get_knn_curve(all_features, k_range=None, chance_adjust=False):
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
        constistencies.append(knn_consistency(knns[x], knns[y], ks=k_range, chance_adjust=chance_adjust))
    constistencies = np.column_stack(constistencies)

    mean_curve = constistencies.mean(axis=1)
    std_curve = constistencies.std(axis=1, ddof=1)

    return (mean_curve, std_curve, constistencies), k_range

def get_ari_curve(all_features, n_clusters_range=None, seeds=[42], pairing="features"):
    def _cluster(points, n_clusters, seed):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        labels = kmeans.fit_predict(points)
        return labels

    def _averaged_ari(labels_grid):
        """
        labels_grid: list of groups, each group a list of label arrays to pair up
        (pairwise ARI computed within each group). Returns the mean/std of those
        pairwise ARIs, averaged across groups.
        """
        n_inner = len(labels_grid[0])
        x_idcs, y_idcs = np.tril_indices(n_inner, k=-1)

        group_means, group_stds = [], []
        for group in labels_grid:
            aris = np.array([
                adjusted_rand_score(group[x], group[y])
                for x, y in zip(x_idcs, y_idcs)
            ])
            group_means.append(aris.mean())
            group_stds.append(aris.std(ddof=1))

        return np.mean(group_means), np.mean(group_stds)

    if n_clusters_range is None:
        n_clusters_range = np.unique(np.geomspace(5, 100, num=20, dtype=int))

    mean_curve, std_curve = [], []

    for n in n_clusters_range:
        # labels_grid[i][j] = labels for seeds[i], all_features[j]
        labels_grid = [
            [_cluster(features, n, seed) for features in all_features]
            for seed in seeds
        ]

        if pairing == "seeds":
            # transpose so each group is the seeds for one feature set
            labels_grid = list(map(list, zip(*labels_grid)))
        # pairing="features": groups are already features-within-a-seed

        mean_ari, std_ari = _averaged_ari(labels_grid)
        mean_curve.append(mean_ari)
        std_curve.append(std_ari)

    return (mean_curve, std_curve), n_clusters_range

def cka(X, Y):
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)

    xy = (Xc.T @ Yc).norm(p='fro')
    xx = (Xc.T @ Xc).norm(p='fro')
    yy = (Yc.T @ Yc).norm(p='fro')

    return xy**2 / (xx * yy)

def get_cka(all_features):
    x_idcs, y_idcs = np.tril_indices(len(all_features), k=-1)
    ckas = np.array(
        [cka(all_features[x], all_features[y]) 
        for x, y in zip(x_idcs, y_idcs)]
    )
    cka_mean = ckas.mean()
    cka_std = ckas.std(ddof=1)
    return cka_mean, cka_std

def rsa(X, Y, use_ranks=False, num_subsample=None):
    X = X.double()
    Y = Y.double()

    n, d = X.shape
    i, j = torch.tril_indices(n, n, offset=-1)
    rmd_x = torch.cdist(X, X)[i, j]
    rmd_y = torch.cdist(Y, Y)[i, j]

    if num_subsample is not None:
        m = rmd_x.shape[0]
        idcs = torch.randint(0, m, (num_subsample, ))
        rmd_x = rmd_x[idcs]
        rmd_y = rmd_y[idcs]


    if use_ranks:
        rmd_x = rmd_x.argsort().argsort().double()
        rmd_y = rmd_y.argsort().argsort().double()

    xc = rmd_x - rmd_x.mean()
    yc = rmd_y - rmd_y.mean()

    return (xc @ yc) / ((xc @ xc) * (yc @ yc)).sqrt()

def get_rsa(all_features, use_ranks=False, num_subsample=None):
    x_idcs, y_idcs = np.tril_indices(len(all_features), k=-1)
    rsas = np.array(
        [rsa(all_features[x], all_features[y], use_ranks=use_ranks, num_subsample=num_subsample) 
        for x, y in zip(x_idcs, y_idcs)]
    )
    rsa_mean = rsas.mean()
    rsa_std = rsas.std(ddof=1)
    return rsa_mean, rsa_std

def get_metrics(all_features, k_range, n_range, chance_adjust=False, use_ranks=True, num_subsample=10_000_000):
    metrics = {}

    metrics['kNN consistency'] = get_knn_curve(all_features, k_range, chance_adjust=chance_adjust)
    metrics['ARI'] = get_ari_curve(all_features, n_range)
    metrics['CKA'] = get_cka(all_features)
    metrics['RSA'] = get_rsa(all_features, use_ranks=use_ranks, num_subsample=num_subsample)

    return metrics
    