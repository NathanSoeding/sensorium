from functools import partial
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import modules
from neuralpredictors.measures import ZIGLoss
from neuralpredictors.layers.encoders import ZIGEncoder
from neuralpredictors.layers.encoders.mean_variance_functions import fitted_zig_mean
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    LongCycler,
)
from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss, get_mean_q, get_mean_theta
from ..utility.utils import set_random_seed

import wandb
from sklearn.cluster import KMeans
from torch.nn import KLDivLoss


def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    loss_type="poisson",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    track_training=False,
    log_smoothness=False,
    detach_core=False,
    use_wandb=True,
    wandb_project=None,
    wandb_config=None,
    wandb_name="",
    # Nina's params
    include_kldivergence=True,
    cluster_number=10,
    alpha=1.0,
    dec_starting_epoch=1,
    kmeans_init=20,
    base_multiplier=4e3,
    learn_alpha=False,
    exponent=2,
    include_mixingcoefficients=False,
    # end of Nina's params
    # dedup-aware DEC clustering (see sensorium.utility.dedup)
    dedup_mode=None,
    dedup_info=None,
    # whiten readout features before DEC clustering (see model.whitener)
    kl_after_whitening=False,
    **kwargs
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        track_training: whether to track and print out the training progress
        **kwargs:

    Returns:

    """
    is_zig_model = isinstance(model, ZIGEncoder)
    assert loss_type in ("poisson", "zig"), f"loss_type must be 'poisson' or 'zig', got {loss_type!r}"
    if not is_zig_model:
        assert loss_type == "poisson", "loss_type='zig' requires a ZIG-style model (readout_type zig/zig_factorized)"
    assert not (
        include_kldivergence and is_zig_model
    ), "include_kldivergence and ZIG-style readouts (zig/zig_factorized) cannot both be used"
    if kl_after_whitening:
        assert getattr(model, "whitener", None) is not None, (
            "kl_after_whitening=True requires the model to have a whitener "
            "(train with whitener=True / without --disable_whitener)"
        )
# --------------
# Nina's code from https://github.com/Nisone2000/DECEMber/blob/main/sensorium/training/trainers.py
# --------------
    def get_multiplier(epoch, base_multiplier=4e3):
        """Multiplier to scale KL loss in same order of magnitude as main loss
        To avoid hard peek aat starting epoch we include a warm-up phase s.t. the loss can increase slower
        """
        if epoch < dec_starting_epoch:
            return 0
        else:
            return base_multiplier

    def target_distribution(batch: torch.Tensor, exponent=exponent) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        p_ij = (q_ij^2/f_j) / sum_j'(q_ij'^2/f_j')  f_j =sum_i(q_ij)

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**exponent) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def _canonicalize_features(features, outdims):
        """
        Nina's DEC-clustering code assumes readout.features (after squeeze) is (channels,
        outdims) -- true for FullGaussian2d but not Factorized2d, whose features are stored
        (outdims, channels). Returns features as (outdims, channels) regardless of the
        underlying readout's storage convention.
        """
        features = features.squeeze()
        return features if features.shape[0] == outdims else features.T

    def _maybe_whiten(features):
        """features: (outdims, channels). When kl_after_whitening, applies the same
        weight-whitening transform used for the whitened feature regularizer
        (model.whitener.transform_weights), so DEC clustering operates in the whitener's
        decorrelated channel space instead of the raw readout feature space."""
        if not kl_after_whitening:
            return features
        return model.whitener.transform_weights(features.T).T

    def pool_features(features, info, mode):
        """Pools a session's per-neuron feature matrix (n_neurons, D) down to one row per
        dedup group (n_groups, D), so duplicate z-plane copies of the same physical cell don't
        each get an independent vote in the DEC clustering computation.

        mode='mean': differentiable group-mean (gradient split evenly across group members).
        mode='random_representative': re-sampled every call, one random member stands in for
        its group (vectorized gather, no python loop).
        """
        if mode == "mean":
            pooled = torch.zeros(info["n_groups"], features.shape[1], device=features.device, dtype=features.dtype)
            pooled.index_add_(0, info["group_id"], features)
            return pooled / info["group_sizes"].unsqueeze(1).to(pooled.dtype)
        elif mode == "random_representative":
            col = (torch.rand(info["n_groups"], device=features.device) * info["group_sizes"]).long()
            chosen = info["group_members_padded"][torch.arange(info["n_groups"], device=features.device), col]
            return features[chosen]
        else:
            raise ValueError(f"Unknown dedup_mode: {mode}")

    def soft_assignments_mult(encoded_features, cluster_centers, sigma, alpha, p=1, mixing_coefficients=None):
        """Calculates the q_ij as the t mixture components. Moves to log space to avoid numerical issues."""
        sigma_inv = 1.0 / sigma  # (K, D)
        diff = encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)  # (N, K, D)
        norm_sigma = torch.sum(diff * sigma_inv * diff, dim=2)  # (N, K)
        det = torch.sum(torch.log(sigma), dim=1)  # log(det) since sigma is diagonal
        log_gamma_top = torch.lgamma((alpha + p) / 2)
        log_gamma_bottom = torch.lgamma(alpha / 2)
        # Log-density formula for multivariate Student-t
        if mixing_coefficients is None:
            log_pdf = (
                log_gamma_top
                - log_gamma_bottom
                - 0.5 * det
                - (p / 2) * torch.log(alpha * torch.pi)
                - ((alpha + p) / 2) * torch.log(1 + (norm_sigma / alpha))
            ) 
        else:
            log_pdf = (
                log_gamma_top
                - log_gamma_bottom
                - 0.5 * det
                - (p / 2) * torch.log(alpha * torch.pi)
                - ((alpha + p) / 2) * torch.log(1 + (norm_sigma / alpha))
            ) + torch.log(mixing_coefficients.squeeze())  
        log_assignments = log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True)
        return torch.exp(log_assignments)  # Convert log-assignments to probabilities

    def EM_t_mult(features, resp, cluster_centers, sigma, alpha, d=1):
        "Does EM updates of centers, shape matrix and mixing coefficients for multivariate t-distribution"
        sigma_inv = 1.0 / sigma  # (K,)
        diff = features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)
        norm_sigma = torch.sum((diff**2 * sigma_inv), 2)
        u = ((alpha + d) / (alpha + norm_sigma)).detach()  # ccalculate U shape(N,K)
        mixing_coefficients = 1/features.shape[1] * torch.sum(resp, dim=0, keepdim=True).T.detach() # (K,)
       
        """ M step """
        numerator = torch.matmul(features, resp * u).T.detach()
        denominator = torch.sum(resp * u, dim=0, keepdim=True).T.detach()
        cluster_centers = numerator / denominator
        weighted_sq_diff = resp.unsqueeze(2) * u.unsqueeze(2) * (diff**2)  # (N, K, D)
        numerator = weighted_sq_diff.sum(dim=0)  # (K,D)
        denominator = torch.sum(resp, dim=0, keepdim=True)  # (K,)
        sigma = (numerator / denominator.T).detach()
        sigma = torch.clamp(sigma, min=1e-4, max=1e4)

        return cluster_centers, sigma, mixing_coefficients

# --------------
    if wandb_project and use_wandb:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
            name=wandb_name,
        )
        if is_zig_model:
            # one-time run metadata (not a per-epoch metric, hence `summary` not `log`):
            # sanity-check the precomputed, frozen per-neuron loc/k that gamma_params_from_data.py produced.
            with torch.no_grad():
                all_loc = torch.cat([model.loc_nl(v).flatten() for v in model.logloc.values()])
                all_k = torch.cat([model.k_nl(v).flatten() for v in model.logk.values()])
            wandb.run.summary.update({
                "gamma_params/loc_mean": all_loc.mean().item(),
                "gamma_params/loc_median": all_loc.median().item(),
                "gamma_params/loc_min": all_loc.min().item(),
                "gamma_params/loc_max": all_loc.max().item(),
                "gamma_params/k_mean": all_k.mean().item(),
                "gamma_params/k_median": all_k.median().item(),
                "gamma_params/k_min": all_k.min().item(),
                "gamma_params/k_max": all_k.max().item(),
            })

    def full_objective(model, dataloader, data_key, *args, **kwargs):

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        core_reg = int(not detach_core) * model.core.regularizer()
        whitener = getattr(model, "whitener", None)
        reg_result = model.readout.regularizer(data_key, whitener=whitener)
        if isinstance(reg_result, tuple):
            readout_reg, readout_reg_components = reg_result
        else:
            readout_reg, readout_reg_components = reg_result, {"feature": reg_result, "smoothness": 0}

        imgs = args[0].to(device)
        preds = model(imgs, data_key=data_key, **kwargs)
        targets = args[1].to(device)
        if is_zig_model and loss_type == "zig":
            # full ZIG negative log-likelihood: preds is the (theta, k, loc, q) tuple
            prediction_loss = loss_scale * criterion(targets, preds)
        elif is_zig_model:
            # loss_type == "poisson": Poisson loss on the analytical mean of the predicted
            # ZIG distribution (trains toward the conditional mean, not the full distribution)
            theta, k, loc, q = preds
            prediction_loss = loss_scale * criterion(fitted_zig_mean(theta, k, loc, q), targets)
        else:
            prediction_loss = loss_scale * criterion(preds, targets)
        loss = prediction_loss + core_reg + readout_reg

        return loss, (prediction_loss, core_reg, readout_reg_components)
        
    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    # losses are summed for each batch
    kldiv_criterion = KLDivLoss(
        size_average=False
    )  # losses are summed for each minibatch

    criterion = (
        ZIGLoss(avg=avg_loss)
        if (is_zig_model and loss_type == "zig")
        else getattr(modules, loss_function)(avg=avg_loss)
    )
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    if learn_alpha:
        raw_alpha = torch.nn.Parameter(
            torch.tensor(1.0, device=device, requires_grad=True)
        )
        optimizer = torch.optim.Adam(list(model.parameters()) + [raw_alpha], lr=lr_init)
    else:
        alpha = torch.tensor(alpha, device=device, requires_grad=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if wandb_project and use_wandb:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if is_zig_model:
            tracker_dict["mean_q"] = partial(get_mean_q, model, dataloaders["validation"], device=device)
            tracker_dict["mean_theta"] = partial(get_mean_theta, model, dataloaders["validation"], device=device)
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    kldiv_list = []
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        if include_kldivergence and epoch == dec_starting_epoch:
            # TODO: include hidden dimension
            cluster_centers_list = []
            kmeans = KMeans(
                n_clusters=cluster_number, n_init=kmeans_init, random_state=seed
            )
            feature_list = []
            # form initial cluster centres
            with torch.no_grad():
                for i,(k,readout) in enumerate(model.readout.items()):
                    features = _canonicalize_features(readout.features.detach(), readout.outdims)
                    features = _maybe_whiten(features)
                    if dedup_mode is not None:
                        features = pool_features(features, dedup_info[k], dedup_mode)
                    feature_list.append(features.cpu().numpy())

                features = np.vstack(feature_list)
                predicted = kmeans.fit_predict(features)
            cluster_centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            p = features.shape[1]
            sigma = torch.zeros((cluster_number, p), device=device)
            for k in range(cluster_number):
                cluster_points = torch.from_numpy(features[predicted == k]).to(
                    device
                )
                if len(cluster_points) > 1:

                    sigma[k] = (
                        torch.var(cluster_points, dim=0, unbiased=True) + 1e-6
                    )
                else:
                    sigma[k] = torch.full_like(cluster_points[0], 1e-6)
            mixing_coefficients = torch.ones(cluster_number, device=device, requires_grad=False) / cluster_number

            if learn_alpha:
                raw_alpha.data = torch.tensor(1.0, dtype=torch.float, device=device)

        model.train()
        epoch_loss_main = 0.0
        epoch_loss_core_reg = 0.0
        epoch_loss_feature_reg = 0.0
        epoch_loss_smoothness_reg = 0.0
        epoch_loss_kldiv = 0
        epoch_loss_kldiv_without_scaling = 0
        batch_count = 0
        kldiv_step_count = 0


        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            loss, loss_components = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core
            )
            loss.backward()

            pred_loss, core_reg, readout_reg_components = loss_components
            with torch.no_grad():
                epoch_loss_main += pred_loss.item()
                epoch_loss_core_reg += core_reg.item()
                epoch_loss_feature_reg += readout_reg_components['feature']
                if log_smoothness:
                    epoch_loss_smoothness_reg += readout_reg_components['smoothness']
                batch_count += 1

            if (batch_no + 1) % optim_step_count == 0:
                # TODO maybe remove the hidden dimensions
                if include_kldivergence and epoch >= dec_starting_epoch:
                    kldiv_loss = torch.zeros(1).to(device)
                    feature_list = []
                    for i, (k, readout) in enumerate(model.readout.items()):
                        features = _canonicalize_features(readout.features, readout.outdims)
                        features = _maybe_whiten(features)
                        if dedup_mode is not None:
                            features = pool_features(features, dedup_info[k], dedup_mode)
                        feature_list.append(features.T)

                    # features_subset = torch.cat(features_subset, dim=1)
                    feature_list = torch.cat(feature_list, dim=1)
                    if learn_alpha:
                        alpha = torch.nn.functional.softplus(raw_alpha) + 0.1
                    if include_mixingcoefficients:
                        q = soft_assignments_mult(
                            feature_list, cluster_centers, sigma, alpha, p, mixing_coefficients
                        )
                    else:
                        q = soft_assignments_mult(
                            feature_list, cluster_centers, sigma, alpha, p
                        )

                    q = q.clamp(min=1e-8)
                    target = target_distribution(q, exponent)
                    target = target.clamp(min=1e-8)

                    kldiv_loss = get_multiplier(epoch, base_multiplier) * (
                        kldiv_criterion(q.log(), target)
                    )
                    kldiv_loss.backward()
                    epoch_loss_kldiv += kldiv_loss.detach()
                    epoch_loss_kldiv_without_scaling += (
                        kldiv_loss.detach() / get_multiplier(epoch, base_multiplier)
                    )
                    kldiv_step_count += 1
                    # epoch_loss += kldiv_loss.detach()

                    with torch.no_grad():
                        cluster_centers_list.append(cluster_centers.cpu().detach())
                        kldiv_list.append(
                            kldiv_loss.cpu() / get_multiplier(epoch, base_multiplier)
                        )

                    cluster_centers, sigma, mixing_coefficients = EM_t_mult(
                        feature_list, q, cluster_centers, sigma, alpha, p
                    )

                optimizer.step()
                optimizer.zero_grad()

        # Calculate average epoch losses
        if batch_count > 0:
            epoch_loss_main /= batch_count
            epoch_loss_core_reg /= batch_count
            epoch_loss_feature_reg /= batch_count
            epoch_loss_smoothness_reg /= batch_count
        if kldiv_step_count > 0:
            epoch_loss_kldiv /= kldiv_step_count
            epoch_loss_kldiv_without_scaling /= kldiv_step_count

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        model.eval()

        # Print and log metrics after each epoch
        if tracker is not None:
            if wandb_project and use_wandb:
                wandb_dict = {
                    "Main loss": epoch_loss_main,
                    "Core Regularizers": epoch_loss_core_reg,
                    "Feature Regularizers": epoch_loss_feature_reg,
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                }
                if log_smoothness:
                    wandb_dict["Smoothness Regularizers"] = epoch_loss_smoothness_reg
                if include_kldivergence:
                    wandb_dict["KL Divergence Loss"] = float(epoch_loss_kldiv)
                    wandb_dict["KL Divergence Loss (unscaled)"] = float(epoch_loss_kldiv_without_scaling)

            # Log validation metrics from tracker
            for key in tracker.log.keys():
                key_val = tracker.log[key][-1]
                if wandb_project and use_wandb:
                    wandb_dict[f"val/{key}"] = key_val
            
            # Log to wandb
            if wandb_project and use_wandb:
                wandb.log(wandb_dict, step=epoch)

    ##### Model evaluation ####################################################################################################
    model.eval()
    if include_kldivergence:
        predicted_list = []
        for i,(k, readout) in enumerate(model.readout.items()):
            features = _canonicalize_features(readout.features.detach(), readout.outdims)
            features = _maybe_whiten(features)
            if dedup_mode is not None:
                # Assign a single label per dedup group (from its pooled/representative
                # features), then broadcast that same label to every neuron in the group --
                # so duplicates share a label by construction, not just by emergent training
                # convergence. group_id maps each neuron -> its group index (see build_dedup_tensors),
                # so group_labels[group_id] re-expands group-level labels back to per-neuron order.
                pooled = pool_features(features, dedup_info[k], dedup_mode).T
                if include_mixingcoefficients:
                    q_groups = soft_assignments_mult(pooled, cluster_centers, sigma, alpha, p, mixing_coefficients)
                else:
                    q_groups = soft_assignments_mult(pooled, cluster_centers, sigma, alpha, p)
                group_labels = q_groups.max(1)[1]
                predicted_list.append(group_labels[dedup_info[k]["group_id"]])
            else:
                features = features.T
                if include_mixingcoefficients:
                    q = soft_assignments_mult(features, cluster_centers, sigma, alpha, p, mixing_coefficients)
                else:
                    q = soft_assignments_mult(features, cluster_centers, sigma, alpha, p)
                predicted_list.append(q.max(1)[1])
        predicted = torch.cat(predicted_list)
        cluster_centers_list.append(cluster_centers.cpu().detach().numpy())
        cluster_centers_np = np.array(cluster_centers_list)
        sigma_np = sigma.cpu().detach().numpy()
        mixing_coefficients = mixing_coefficients.cpu().detach().numpy() if include_mixingcoefficients else None

    if tracker is not None:
        tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training and tracker is not None else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)

    if include_kldivergence:
        output['cluster_centers_np'] = cluster_centers_np
        output['sigma_np'] = sigma_np
        output['mixing_coefficients'] = mixing_coefficients
        output['predicted'] = predicted

    # Log final metrics
    if wandb_project and use_wandb:
        wandb.log({
            "final/validation_corr_mean": score,
        })
        wandb.finish()

    return score, output, model.state_dict()
