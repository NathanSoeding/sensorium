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

def get_filtered_correlations(
    model, dataloaders, tier=None, device='cpu', as_dict=False, per_neuron=True, neuron_idcs=None
):
    ''' Applies 'stop_closure' per neuron and filters specific neurons to compute the final score. '''
    

    correlations = {}
    dl = dataloaders[tier] if tier is not None else dataloaders 
    for k, v in dl.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        # Filter neurons
        if neuron_idcs is not None:
            target = target[:, neuron_idcs[k]]
            output = output[:, neuron_idcs[k]]

        correlations[k] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


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
    train_neurons=None,
    validation_neurons=None,

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

    def full_objective(model, dataloader, data_key, neuron_idcs, *args, **kwargs):

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
        if neuron_idcs is not None:
            preds = preds[:, neuron_idcs[data_key]]
            targets = targets[:, neuron_idcs[data_key]]
        
        loss = loss_scale * criterion(preds, targets) + regularizers
        return loss

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss)

    assert (
        (validation_neurons is None) == (train_neurons is None)
    ), "if one of val_neurons or train_neurones is defined the other is also required"

    if validation_neurons is None:
        stop_closure = partial(
            getattr(scores, stop_function),
            dataloaders=dataloaders["validation"],
            device=device,
            per_neuron=False,
            avg=True,
        )
    else:
        stop_closure = partial(
            get_filtered_correlations, 
            dataloaders=dataloaders['validation'], 
            device=device, 
            per_neuron=False,
            neuron_idcs=validation_neurons,
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
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if wandb_project:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
        )
        wandb.run.name = wandb_name

    if wandb_project:
        tracker_dict = dict(
            train_correlaitons=partial(
                get_filtered_correlations,
                model,
                dataloaders['validation'],
                device=device,
                per_neuron=False,
                neuron_idcs=train_neurons,
            ),
            val_correlation=partial(
                get_filtered_correlations,
                model,
                dataloaders['validation'],
                device=device,
                per_neuron=False,
                neuron_idcs=validation_neurons,
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

        # print the quantities from tracker
        if wandb_project and tracker is not None:
            log_dict = {}
            for key in tracker.log.keys():
                val = tracker.log[key][-1]
                log_dict[f"val/{key}"] = val
                wandb.log(log_dict, step=epoch)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                train_neurons,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core
            )
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_filtered_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False, neuron_idcs=validation_neurons, 
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)

    if wandb_project:
        wandb.log({
            "final/validation_corr_mean": score,
            "final/validation_corr_all": validation_correlation,
        })

    wandb.finish()

    return score, output, model.state_dict()
