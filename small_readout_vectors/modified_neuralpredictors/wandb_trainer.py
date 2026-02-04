import warnings
from functools import partial
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import modules
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    LongCycler,
)
from nnfabrik.utility.nn_helpers import set_random_seed

from sensorium.utility import scores
from sensorium.utility.scores import get_correlations, get_poisson_loss, model_predictions

import wandb

def avg_weight_divergence(model, dataloaders):
    data_keys = dataloaders['train'].keys()
    diffs = []

    for key in data_keys:
        features = model.readout[key].features
        diffs.append(
            (features[:, :64, :, :] - features[:, 64:, :, :]).flatten().abs()
        )

    diffs = torch.cat(diffs)
    avg_diff = diffs.mean()
    return avg_diff

def barlow_loss_fn(model, data_key):
    feature_emb, _ = model.readout[data_key].bottleneck.get_last_embeds()
    feature_emb = feature_emb.transpose(0, 1)
    n, b, d = feature_emb.shape

    cov = torch.bmm(
        feature_emb.transpose(1, 2), 
        feature_emb
    ) / (b - 1)  # batched matrix mult (one cov per neuron)

    identity = torch.eye(d, device=cov.device)
    barlow_loss = (cov - identity).pow(2).sum()
    print(cov[0])

    return barlow_loss

def topographic_loss_fn(predictions, model, data_key, std_threshold=0.01, eps=1e-4, k=None):
    low_std_filter = predictions.std(dim=1) > std_threshold
    predictions = predictions[low_std_filter]
    
    embeds = model.readout[data_key].features.squeeze().T  # N x d
    embeds = embeds[low_std_filter]  
    
    i, j = np.tril_indices(predictions.shape[0], k=-1)

    pred_corr = torch.corrcoef(predictions)
    pred_corr = pred_corr[i, j]  # Take every pair once

    dist = torch.cdist(embeds, embeds, p=2)
    dist = dist[i, j]  # Take every pair once
    #dists_inv = 1 / (1 + dists)
    
    if k is not None:
        _, pred_corr_idcs = torch.topk(pred_corr, k, largest=True)
        _, dist_idcs = torch.topk(dist, k, largest=False)
        union_idcs = torch.unique(
            torch.cat([pred_corr_idcs, dist_idcs]), 
            sorted=False
        )
        pred_corr = pred_corr[union_idcs]
        dist = dist[union_idcs]

    pred_corr_c = pred_corr - pred_corr.mean()
    #dists_inv_0 = dists_inv - dists_inv.mean()
    dist_c = dist - dist.mean()

    #topographic_loss = (
    #    - (pred_corr_0 @ dists_inv_0) 
    #    / (torch.norm(pred_corr_0) * torch.norm(dists_inv_0) + eps)
    #)

    topographic_loss = (
        (pred_corr_c @ dist_c) 
        / (torch.norm(pred_corr_c) * torch.norm(dist_c) + eps)
    )

    return topographic_loss

def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
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
    detach_core=False,
    wandb_project=None,
    wandb_config=None,
    wandb_name="", 
    topographic_loss_w=None,
    topographic_loss_k=None,
    barlow_loss_w=None, 
    use_wandb=True,  # Added parameter to control wandb usage
    optimizer=None, 
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
        use_wandb: whether to use wandb logging (default: True)
        **kwargs:

    Returns:

    """
    
    # Initialize wandb if specified
    if wandb_project and use_wandb:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
            name=wandb_name,
        )

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        topographic_loss = torch.zeros(1).to(device)
        barlow_loss = torch.zeros(1).to(device)

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        regularizers = int(
            not detach_core
        ) * model.core.regularizer() + model.readout.regularizer(data_key) 
        # Here I removed readout regularization for overcompleteness sanity check
        
        imgs = args[0].to(device)
        preds = model(imgs, data_key=data_key, **kwargs)
        targets = args[1].to(device)
        
        poisson_loss = loss_scale * criterion(preds, targets) 
        
        if topographic_loss_w is not None:
            topographic_loss = topographic_loss_w * topographic_loss_fn(preds.T, model, data_key, k=topographic_loss_k)

        if barlow_loss_w is not None:
            barlow_loss = barlow_loss_w * barlow_loss_fn(model, data_key)

        loss = poisson_loss + regularizers + topographic_loss + barlow_loss
        return loss, (poisson_loss, topographic_loss, barlow_loss, regularizers)
    
    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss)

    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    # Initialize tracker for validation metrics
    if wandb_project and use_wandb:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model,
                dataloaders['validation'],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model,
                dataloaders['validation'],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # Variables for tracking training losses
    batch_no_tot = 0
    epoch_loss_total = 0.0
    epoch_loss_main = 0.0
    epoch_loss_topographic = 0.0
    epoch_loss_reg = 0.0
    batch_count = 0

    # train over epochs
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

        # Reset epoch loss accumulators
        epoch_loss_main = 0.0
        epoch_loss_topographic = 0.0
        epoch_loss_barlow = 0.0
        epoch_loss_reg = 0.0
        batch_count = 0

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
            
            poisson_loss, topographic_loss, barlow_loss, regularizers = loss_components
            
            loss.backward()
            
            # Accumulate batch losses for epoch statistics
            with torch.no_grad():
                epoch_loss_total += loss.item()
                epoch_loss_main += poisson_loss.item()
                epoch_loss_topographic += topographic_loss.item()
                epoch_loss_barlow += barlow_loss.item()
                epoch_loss_reg += regularizers.item()
                batch_count += 1
            
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            batch_no_tot += 1

        # Calculate average epoch losses
        if batch_count > 0:
            epoch_loss_total /= batch_count
            epoch_loss_main /= batch_count
            epoch_loss_topographic /= batch_count
            epoch_loss_barlow /= batch_count
            epoch_loss_reg /= batch_count

        # Execute callback function if passed in keyword args
        if cb is not None:
            cb()

        # Print and log metrics after each epoch
        if tracker is not None:
            model.eval()
            if wandb_project and use_wandb:
                wandb_dict = {
                    "Epoch Train loss poisson": epoch_loss_main,
                    "Epoch Train loss regularizers": epoch_loss_reg,
                    "Epoch Train loss topographic": epoch_loss_topographic / topographic_loss_w if topographic_loss_w is not None else 0,
                    "Epoch Train loss ratio": epoch_loss_main / epoch_loss_topographic if topographic_loss_w is not None else 0, 
                    "Epoch Train loss barlow": epoch_loss_barlow, 
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                }
            
            if verbose:
                print("=======================================")
                print(f"Epoch {epoch}:")
                print(f"  Train Loss Total: {epoch_loss_total:.4f}")
                print(f"  Train Loss Main (Poisson): {epoch_loss_main:.4f}")
                print(f"  Train Loss Regularizers: {epoch_loss_reg:.4f}")
                if topographic_loss_w is not None:
                    print(f"  Train Loss Topographic: {epoch_loss_topographic:.4f}")
            
            # Log validation metrics from tracker
            for key in tracker.log.keys():
                key_val = tracker.log[key][-1]
                if wandb_project and use_wandb:
                    wandb_dict[f"val/{key}"] = key_val
                if verbose:
                    print(f"  Validation {key}: {key_val:.4f}")
            
            # Log to wandb
            if wandb_project and use_wandb:
                wandb.log(wandb_dict, step=epoch)

    ##### Model evaluation ####################################################################################################
    model.eval()
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

    # Log final metrics
    if wandb_project and use_wandb:
        wandb.log({
            "final/validation_corr_mean": score,
            "final/validation_corr_all": validation_correlation,
        })
        wandb.finish()

    return score, output, model.state_dict()