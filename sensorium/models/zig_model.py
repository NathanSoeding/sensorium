import os

import numpy as np
import torch
from torch import nn

from neuralpredictors.utils import get_module_output
from neuralpredictors.layers.encoders import ZIGEncoder
from neuralpredictors.layers.shifters import MLPShifter, StaticAffine2dShifter
from neuralpredictors.layers.whiteners import Whitener
from neuralpredictors.layers.cores import Stacked2dCore

from ..utility.utils import set_random_seed, get_dims_for_loader_dict
from .readouts import MultipleGeneralizedFullGaussian2d
from .utility import prepare_grid


def _loc_to_logloc(loc):
    """Invert ZIGEncoder.loc_nl: loc_nl(logloc) = exp(logloc)."""
    return torch.log(loc)


def _k_to_logk(k, offset):
    """
    Invert ZIGEncoder.k_nl in the `zero_thresholds is not None` branch:
    k_nl(logk) = elu(logk) + 1 + offset
    """
    v = k - 1 - offset
    return torch.where(v >= 0, v, torch.log(v + 1))


def _load_gamma_params(gamma_params_dir, data_key, n_neurons):
    path = os.path.join(gamma_params_dir, f"{data_key}_gamma_params.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No precomputed gamma params found at {path}. Run "
            "sensorium/utility/gamma_params_from_data.py first (with a matching --output_dir)."
        )
    data = np.load(path)
    loc, k = data["loc"], data["k"]
    if len(loc) != n_neurons or len(k) != n_neurons:
        raise ValueError(
            f"Gamma params for {data_key} have {len(loc)} neurons but the model expects {n_neurons} "
            "(neuron-subsetting mismatch between gamma_params_from_data.py and this training run?)."
        )
    return torch.from_numpy(loc).float(), torch.from_numpy(k).float()


def stacked_core_zig_gauss_readout(
    dataloaders,
    seed,
    gamma_params_dir,
    hidden_channels=32,
    input_kern=13,
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    nonlinearity_type='AdaptiveELU',
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    batch_norm_scale=[],
    track_running_stats=False,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    use_avg_reg=False,
    init_mu_range=0.2,
    init_sigma=1.0,
    readout_bias=True,
    stack=None,
    depth_separable=False,
    linear=False,
    gauss_type="full",
    grid_mean_predictor=None,
    attention_conv=False,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    init_gain=1.0,
    gamma_shifter=0,
    shifter_bias=True,
    hidden_padding=None,
    core_bias=True,
    feature_reg_weight=4,
    regularizer_type="adaptive_log_norm",
    gamma_sigma=0.25,
    zig_offset=1.0e-6,
    whitener=None,
    whitener_momentum=0.003,
):
    """
    Model class of a Stacked2dCore (from neuralpredictors) and a GeneralizedFullGaussianReadout2d
    readout with 2 image-dependent outputs (q, theta), fused into a ZIGEncoder. `loc` and `k` are
    estimated per neuron from data (see sensorium/utility/gamma_params_from_data.py) and held frozen
    (not fine-tuned).

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        gamma_params_dir: directory containing one `<data_key>_gamma_params.npz` file per session,
            produced by sensorium/utility/gamma_params_from_data.py
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            GeneralizedFullGaussianReadout2d in neuralpredictors.layers.readouts

    Returns: An initialized ZIGEncoder model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    batch = next(iter(list(dataloaders.values())[0]))
    in_name, out_name = (
        list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
    )

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)
    grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(grid_mean_predictor, dataloaders)

    core = Stacked2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        nonlinearity_type=nonlinearity_type,
        bias=core_bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        track_running_stats=track_running_stats,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        depth_separable=depth_separable,
        linear=linear,
        attention_conv=attention_conv,
        hidden_padding=hidden_padding,
        use_avg_reg=use_avg_reg,
        batch_norm_scale=batch_norm_scale,
    )

    if whitener is True:
        whitener = Whitener(
            model_dim=hidden_channels,
            momentum=whitener_momentum,
        )

    in_shapes_dict = {
        k: get_module_output(core, v[in_name])[1:]
        for k, v in session_shape_dict.items()
    }

    readout = MultipleGeneralizedFullGaussian2d(
        in_shape_dict=in_shapes_dict,
        loader=dataloaders,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        feature_reg_weight=feature_reg_weight,
        regularizer_type=regularizer_type,
        gamma_sigma=gamma_sigma,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        inferred_params_n=2,  # readout predicts q, then theta (see forward_base slicing order)
    )

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
                bias=shifter_bias,
                init_gain=init_gain,
            )
        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    data_keys = list(dataloaders.keys())
    model = ZIGEncoder(
        core=core,
        readout=readout,
        shifter=shifter,
        whitener=whitener,
        # dummy per-session scalar: only used to flip k_nl's internal branch to the
        # `k > offset` floor (instead of `k > 1.1`) since MLE-fit k can be < 1.
        # The actual per-neuron loc/k are loaded and frozen in below.
        zero_thresholds={data_key: 1.0 for data_key in data_keys},
        loc_image_dependent=False,
        q_image_dependent=True,
        theta_image_dependent=True,
        k_image_dependent=False,
        offset=zig_offset,
    )

    for data_key in data_keys:
        loc, k = _load_gamma_params(gamma_params_dir, data_key, n_neurons_dict[data_key])
        model.logloc[data_key] = nn.Parameter(_loc_to_logloc(loc)[None, :], requires_grad=False)
        model.logk[data_key] = nn.Parameter(_k_to_logk(k, model.offset)[None, :], requires_grad=False)

    return model
