<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

# SENSORIUM 2022 Competition

![plot](figures/Fig1.png)
SENSORIUM is a competition on predicting large scale mouse primary visual cortex activity. We will provide large scale datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs of natural stimuli and recorded neuronal responses, and submit the predicted responses to a set of test images for which responses are withheld.

Join our challenge and compete for the best neural predictive model!

For more information about the competition, vist our [website](https://sensorium2022.net/).

Have a look at our [White paper on arXiv](https://arxiv.org/abs/2206.08666), which describes the dataset and competition in detail.

# Starter-kit

## Repository setup

Clone both repositories side-by-side:

git clone --branch whitened_zig git@github.com:NathanSoeding/sensorium.git  
git clone --branch cleanup-from-upstream git@github.com:NathanSoeding/neuralpredictors.git  

Directory structure:  

parent/  
├── sensorium/  
└── neuralpredictors/  

Then:  

cd sensorium  
uv venv  
source .venv/bin/activate  
uv sync  
uv pip install -e .  

Start a run:  
cd training/  
python launcher.py --path runs --name test  


## ZIG setup 

The ZIG here is a Zero-Inflated Gamma: a mixture model for a response that's mostly near-zero (no event) with an occasional continuous positive "burst." You can read
the meaning of each parameter straight off ZIGLoss.get_slab_logl (zero_inflated_losses.py:93-103) and fitted_zig_mean (mean_variance_functions.py:5-9):

- q — mixing weight, P(response is "active"/slab) for this trial. 1-q is the mass on the "spike" (near-zero/no-response) part. This is literally "did the neuron
respond to this stimulus at all" → directly driven by the image.
- theta — Gamma scale. Combined with k, the slab mean is k*theta + loc, so theta sets the magnitude of the response when the neuron is active. This is the "how big was
the response" parameter → directly driven by the image/tuning.
- k — Gamma shape. Controls the dispersion/skew of the nonzero-response distribution around that mean — i.e. how noisy/variable the response is trial-to-trial given
that the neuron fired, not how big the response is. More a property of the noise structure than of stimulus content.
- loc — the location shift, i.e. the boundary between "zero" and "nonzero": zero_mask = target <= loc in the loss. This is effectively the neuron's noise floor (from
calcium-imaging/deconvolution noise), not something the stimulus should move around.

### How to train ZIG model 

**Step 1**
```
python sensorium/utility/gamma_params_from_data.py \
```
or 
```
python sensorium/utility/gamma_params_from_data.py \
    --loc_method quantile --loc_quantile 0.01 \
    --output_dir sensorium/notebooks/data/gamma_params
```
Add `--more_data` here if you plan to train on the extra 7 sessions too (must match whatever you pass to train.py in step 2 — the `.npz` files are looked up by session data_key, and a missing one raises immediately). This writes one `sensorium/notebooks/data/gamma_params/<data_key>_gamma_params.npz` per session.

**Step 2**
```
cd training
python launcher.py --path runs --name zig_test \
    --readout_type zig \
    --use_zig_loss \
    --gamma_params_dir ../data/gamma_params
```