from functools import partial
import warnings
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import modules
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    JointCycler,
    LongCycler,
)
from neuralpredictors.layers.encoders import decov_variance_floor_loss, select_decov_target_vecs
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
    detach_core=False,
    joint_cycler=False,
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
        joint_cycler: if True, each training step draws one batch from *every* session at once
            (via JointCycler) instead of one session per step (LongCycler). The DeCov /
            variance-floor penalties (model.decov_weight / model.variance_floor_weight) are then
            computed once per step from a single covariance pooled across all sessions' feature
            vectors, with gradients flowing through every session in that step -- instead of one
            independent per-session covariance per step, as happens by default. If
            model.whitener is also in 'batch' mode, the whitening itself is pooled the same way:
            all 7 sessions are whitened with one shared mean/covariance computed fresh from their
            concatenated raw feature vectors (see Whitener.joint_whiten_batch), instead of each
            session whitening itself from its own batch. Requires `dataloaders["train"]` to have
            loaders of comparable batch size for all sessions.
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
        out = model.readout.regularizer(data_key, whitener=model.whitener)
        if type(out) == tuple:
            readout_reg, readout_reg_components = out
        else:
            readout_reg = out
            readout_reg_components = {'feature': readout_reg.item()}

        imgs = args[0].to(device)
        preds = model(imgs, data_key=data_key, **kwargs)
        targets = args[1].to(device)
        prediction_loss = loss_scale * criterion(preds, targets)

        variance_floor_reg = model.last_variance_floor_loss
        decov_reg = model.last_decov_loss

        loss = prediction_loss + core_reg + readout_reg + variance_floor_reg + decov_reg

        return loss, (prediction_loss, core_reg, readout_reg_components, variance_floor_reg, decov_reg)

    def joint_objective(model, dataloader, session_batches, *, detach_core=False):
        # core_reg is data-independent (weight-based), so it must be added once per step, not once
        # per session, or it would silently get scaled by the number of sessions.
        core_reg = int(not detach_core) * model.core.regularizer()

        def session_loss_scale_and_readout_reg(data_key, n_images):
            loss_scale = (
                np.sqrt(len(dataloader[data_key].dataset) / n_images) if scale_loss else 1.0
            )
            out = model.readout.regularizer(data_key, whitener=model.whitener)
            if type(out) == tuple:
                readout_reg, components = out
            else:
                readout_reg = out
                components = {'feature': readout_reg.item()}
            return loss_scale, readout_reg, components

        total_prediction_loss = 0.0
        total_readout_reg = 0.0
        readout_reg_components = {}
        target_vecs_per_session = []

        joint_batch_whitening = model.whitener is not None and model.whitener.mode == "batch"

        if not joint_batch_whitening:
            for data_key, data in session_batches:
                batch_args = list(data)
                batch_kwargs = data._asdict() if not isinstance(data, dict) else data

                loss_scale, readout_reg, session_readout_reg_components = session_loss_scale_and_readout_reg(
                    data_key, batch_args[0].shape[0]
                )

                imgs = batch_args[0].to(device)
                preds, feature_vecs = model(
                    imgs, data_key=data_key, return_vec=True, detach_core=detach_core, **batch_kwargs
                )
                targets = batch_args[1].to(device)
                prediction_loss = loss_scale * criterion(preds, targets)

                total_prediction_loss = total_prediction_loss + prediction_loss
                total_readout_reg = total_readout_reg + readout_reg
                for k, v in session_readout_reg_components.items():
                    v = v.item() if torch.is_tensor(v) else v
                    readout_reg_components[k] = readout_reg_components.get(k, 0) + v

                # Selected the same way as inside FiringRateEncoder.forward, but pooled across
                # sessions below instead of turned into a loss per session.
                target_vecs = select_decov_target_vecs(
                    feature_vecs, model.whitener, model.decorrelation_on_raw_features
                )
                target_vecs_per_session.append(target_vecs.flatten(0, 1))
        else:
            # Phase 1: gather every session's raw (pre-whitening) feature vectors, so the
            # whitening below pools statistics across all of them jointly instead of per session
            # (no per-session mu/cov, no EMA/history -- see Whitener.joint_whiten_batch).
            session_state = []
            for data_key, data in session_batches:
                batch_args = list(data)
                batch_kwargs = data._asdict() if not isinstance(data, dict) else data

                loss_scale, readout_reg, session_readout_reg_components = session_loss_scale_and_readout_reg(
                    data_key, batch_args[0].shape[0]
                )

                imgs = batch_args[0].to(device)
                core_out, raw_feature_vecs, shift = model.forward_raw(
                    imgs, data_key=data_key, detach_core=detach_core, **batch_kwargs
                )
                targets = batch_args[1].to(device)

                total_readout_reg = total_readout_reg + readout_reg
                for k, v in session_readout_reg_components.items():
                    v = v.item() if torch.is_tensor(v) else v
                    readout_reg_components[k] = readout_reg_components.get(k, 0) + v

                session_state.append(
                    dict(
                        data_key=data_key,
                        loss_scale=loss_scale,
                        core_out=core_out,
                        raw_feature_vecs=raw_feature_vecs,
                        shift=shift,
                        targets=targets,
                        behavior=batch_kwargs.get("behavior"),
                    )
                )

            # Phase 2: one pooled mean/covariance across all sessions' raw feature vectors,
            # with full gradient -- then finish each session's forward with its whitened slice.
            whitened_per_session = model.whitener.joint_whiten_batch(
                [s["raw_feature_vecs"] for s in session_state]
            )

            for s, whitened in zip(session_state, whitened_per_session):
                preds, feature_vecs = model.forward_from_features(
                    s["core_out"],
                    data_key=s["data_key"],
                    whitened_feature_vecs=whitened,
                    shift=s["shift"],
                    behavior=s["behavior"],
                    return_vec=True,
                )
                prediction_loss = s["loss_scale"] * criterion(preds, s["targets"])
                total_prediction_loss = total_prediction_loss + prediction_loss

                target_vecs = s["raw_feature_vecs"] if model.decorrelation_on_raw_features else feature_vecs
                target_vecs_per_session.append(target_vecs.flatten(0, 1))

        pooled_target_vecs = torch.cat(target_vecs_per_session, dim=0)
        variance_floor_reg, decov_reg = decov_variance_floor_loss(
            pooled_target_vecs, model.decov_weight, model.variance_floor_weight, model.variance_floor_gamma
        )

        loss = total_prediction_loss + core_reg + total_readout_reg + variance_floor_reg + decov_reg

        return loss, (total_prediction_loss, core_reg, readout_reg_components, variance_floor_reg, decov_reg)

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

    cycler_cls = JointCycler if joint_cycler else LongCycler
    n_iterations = len(cycler_cls(dataloaders["train"]))

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
    if loss_accum_batch_n is None:
        # With joint_cycler, one step already is a full pass over all sessions (unlike LongCycler,
        # where the default here accumulates exactly one such pass before stepping) -- so the
        # matching default is to step every joint step, not every len(sessions) of them.
        optim_step_count = 1 if joint_cycler else len(dataloaders["train"].keys())
    else:
        optim_step_count = loss_accum_batch_n
        if joint_cycler and loss_accum_batch_n != 1:
            warnings.warn(
                f"joint_cycler=True with loss_accum_batch_n={loss_accum_batch_n}: gradients will be "
                f"accumulated over {loss_accum_batch_n} joint steps (i.e. {loss_accum_batch_n} passes "
                "over all sessions, holding all their graphs at once) before each optimizer.step()."
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
        epoch_loss_variance_floor = 0.0
        epoch_loss_decov = 0.0
        epoch_loss_read_regs = {}
        batch_count = 0


        # train over batches
        optimizer.zero_grad()
        for batch_no, batch in tqdm(
            enumerate(cycler_cls(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            if joint_cycler:
                loss, loss_components = joint_objective(
                    model, dataloaders["train"], batch, detach_core=detach_core
                )
            else:
                data_key, data = batch
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

            pred_loss, core_reg, readout_reg_components, variance_floor_reg, decov_reg = loss_components
            with torch.no_grad():
                epoch_loss_main += pred_loss.item()
                epoch_loss_core_reg += core_reg.item()
                epoch_loss_variance_floor += variance_floor_reg.item()
                epoch_loss_decov += decov_reg.item()
                for k, v in readout_reg_components.items():
                    if k not in epoch_loss_read_regs:
                        epoch_loss_read_regs[k] = 0
                    epoch_loss_read_regs[k] += v

                batch_count += 1

            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Calculate average epoch losses
        if batch_count > 0:
            epoch_loss_main /= batch_count
            epoch_loss_core_reg /= batch_count
            epoch_loss_variance_floor /= batch_count
            epoch_loss_decov /= batch_count
            for k, v in epoch_loss_read_regs.items():
                epoch_loss_read_regs[k] /= batch_count

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        whitener_diag = {}
        if getattr(model, "whitener", None) is not None and model.whitener.mode == "batch":
            whitener_diag = {
                "whitener/min": model.whitener.last_min,
                "whitener/max": model.whitener.last_max,
                "whitener/cond": model.whitener.last_cond,
                "whitener/min_abs_eigenvalue": model.whitener.last_min_abs_eigenvalue,
                "whitener/max_abs_eigenvalue": model.whitener.last_max_abs_eigenvalue,
                "whitener/raw_cond": model.whitener.last_raw_cond,
                "whitener/raw_min_abs_eigenvalue": model.whitener.last_raw_min_abs_eigenvalue,
                "whitener/raw_max_abs_eigenvalue": model.whitener.last_raw_max_abs_eigenvalue,
            }

        model.eval()

        # Print and log metrics after each epoch
        if tracker is not None:
            if wandb_project and use_wandb:
                live_whitening_diag = {}
                if getattr(model, "whitener", None) is not None and model.whitener.mode == "batch":
                    with model.whitener.live_stats():
                        live_whitening_correlation = get_correlations(
                            model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
                        )
                    live_whitening_diag = {"val/correlation_live_whitening": float(live_whitening_correlation)}

                wandb_dict = {
                    "Main loss": epoch_loss_main,
                    "Core Regularizers": epoch_loss_core_reg,
                    "Variance Floor Regularizer": epoch_loss_variance_floor,
                    "DeCov Regularizer": epoch_loss_decov,
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    **whitener_diag,
                    **live_whitening_diag,
                }
                for k, v in epoch_loss_read_regs.items():
                    wandb_dict[k] = v
                
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
