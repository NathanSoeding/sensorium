import argparse
import os

import numpy as np
from scipy.stats import gamma

from sensorium.utility.utils import get_data


def estimate_gamma_params(dataloader, loc_method="quantile", loc_quantile=0.01, loc_constant=None, max_value=None):
    """
    Estimate per-neuron ZIG parameters from a single session's train-tier dataloader.

    The dataloader is expected to come from a `normalize=True` dataset pipeline (the
    same one used for training), since the resulting `loc`/`k` are meant to be used
    directly by a model trained on that same normalized response scale.

    Args:
        dataloader: a single-session train-tier torch DataLoader (responses field
            available as `.responses` on a namedtuple batch, or `["responses"]` on a
            dict batch).
        loc_method (str): "quantile" or "constant".
        loc_quantile (float): quantile (over each neuron's nonzero responses) used as
            `loc` when `loc_method="quantile"`.
        loc_constant (float): fixed `loc` value used for every neuron when
            `loc_method="constant"`.
        max_value (float, optional): cap responses above `loc` at this value before
            fitting/computing moments, to guard against rare extreme outliers. `None`
            (default) applies no cap.

    Returns:
        dict with per-neuron arrays "loc", "k", "mean", "std" (each shape (n_neurons,)).
    """
    all_responses = []
    for batch in dataloader:
        responses = batch.responses if hasattr(batch, "responses") else batch["responses"]
        all_responses.append(responses.detach().cpu().numpy())
    responses = np.concatenate(all_responses, axis=0)  # (n_trials, n_neurons)

    n_neurons = responses.shape[1]
    loc = np.zeros(n_neurons, dtype=np.float64)
    k = np.zeros(n_neurons, dtype=np.float64)
    mean = np.zeros(n_neurons, dtype=np.float64)
    std = np.zeros(n_neurons, dtype=np.float64)

    for n in range(n_neurons):
        neuron_responses = responses[:, n]

        if loc_method == "quantile":
            nonzero = neuron_responses[neuron_responses > 0]
            if nonzero.size == 0:
                raise ValueError(f"Neuron {n} has no nonzero responses; cannot estimate a quantile-based loc.")
            neuron_loc = np.quantile(nonzero, loc_quantile)
        elif loc_method == "constant":
            if loc_constant is None:
                raise ValueError("loc_constant must be set when loc_method='constant'")
            neuron_loc = loc_constant
        else:
            raise ValueError(f"Unknown loc_method '{loc_method}', expected 'quantile' or 'constant'")

        active = neuron_responses[neuron_responses > neuron_loc]
        if max_value is not None:
            active = np.minimum(active, max_value)
        if active.size < 2:
            raise ValueError(
                f"Neuron {n} has fewer than 2 responses above loc={neuron_loc}; cannot fit a Gamma "
                "distribution. Consider a smaller --loc_quantile (or a lower --loc_constant)."
            )

        fitted_k, _, _ = gamma.fit(active, floc=neuron_loc)

        loc[n] = neuron_loc
        k[n] = fitted_k
        mean[n] = active.mean()
        std[n] = active.std()

    return {"loc": loc, "k": k, "mean": mean, "std": std}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Estimate per-neuron ZIG loc/k (plus mean/std) from normalized training responses, "
        "saving one .npz per session for use with sensorium.models.zig_model."
    )
    parser.add_argument("--more_data", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.25)

    parser.add_argument("--loc_method", type=str, default="quantile", choices=["quantile", "constant"])
    parser.add_argument("--loc_quantile", type=float, default=0.01)
    parser.add_argument("--loc_constant", type=float, default=None)
    parser.add_argument("--max_value", type=float, default=None)

    parser.add_argument("--output_dir", type=str, default="sensorium/notebooks/data/gamma_params")
    return parser


def main():
    args = get_parser().parse_args()

    basepath = "/srv/user/polina/sensorium/sensorium/notebooks/data/"
    filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file]

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

    dataset_config = {
        "paths": filenames,
        "normalize": True,
        "include_behavior": True,
        "include_eye_position": True,
        "exclude_eye_position_paths": [
            "/srv/user/polina/sensorium/sensorium/notebooks/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"
        ],
        "batch_size": args.batch_size,
        "scale": args.scale,
        "cuda": False,
    }
    dataloaders = get_data("sensorium.datasets.static_loaders", dataset_config)

    os.makedirs(args.output_dir, exist_ok=True)

    for data_key, dataloader in dataloaders["train"].items():
        print(f"Estimating gamma params for {data_key}...")
        params = estimate_gamma_params(
            dataloader,
            loc_method=args.loc_method,
            loc_quantile=args.loc_quantile,
            loc_constant=args.loc_constant,
            max_value=args.max_value,
        )
        out_path = os.path.join(args.output_dir, f"{data_key}_gamma_params.npz")
        np.savez(out_path, **params)
        print(
            f"  saved {out_path} "
            f"(loc mean={params['loc'].mean():.4g}, k mean={params['k'].mean():.4g}, n_neurons={len(params['loc'])})"
        )


if __name__ == "__main__":
    main()
