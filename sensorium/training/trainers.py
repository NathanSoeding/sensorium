from functools import partial
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import modules
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    LongCycler,
)
from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss
from ..utility.utils import set_random_seed

import wandb


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
    log_smoothness=False,
    detach_core=False,
    use_wandb=True,
    wandb_project=None,
    wandb_config=None,
    wandb_name="",
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

    if wandb_project and use_wandb:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
            name=wandb_name,
        )

    def full_objective(model, dataloader, data_key, *args, **kwargs):

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        core_reg = int(not detach_core) * model.core.regularizer()
        readout_reg, readout_reg_components = model.readout.regularizer(data_key, whitener=model.whitener)

        imgs = args[0].to(device)
        preds = model(imgs, data_key=data_key, **kwargs)
        targets = args[1].to(device)
        prediction_loss = loss_scale * criterion(preds, targets)
        loss = prediction_loss + core_reg + readout_reg

        return loss, (prediction_loss, core_reg, readout_reg_components)
        
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
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

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
        model.train()
        epoch_loss_main = 0.0
        epoch_loss_core_reg = 0.0
        epoch_loss_feature_reg = 0.0
        epoch_loss_smoothness_reg = 0.0
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
                optimizer.step()
                optimizer.zero_grad()

        # Calculate average epoch losses
        if batch_count > 0:
            epoch_loss_main /= batch_count
            epoch_loss_core_reg /= batch_count
            epoch_loss_feature_reg /= batch_count
            epoch_loss_smoothness_reg /= batch_count

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
        })
        wandb.finish()

    return score, output, model.state_dict()
