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

git clone --branch cleanup-sensorium git@github.com:NathanSoeding/sensorium.git  
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


## **Competition example notebooks**
We provide notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
<br>[**Dataset tutorial**](notebooks/dataset_tutorial/): Shows the structure of the data and how to turn it into a PyTorch DataLoader.
<br>[**Model tutorial**](notebooks/model_tutorial/): How to train and evaluate our baseline models.
<br>[**Submission tutorial**](notebooks/submission_tutorial/): Use our API to make a submission to our competition.


If you have any questions, feel free to reach out to us (Contact section on our [website](https://sensorium2022.net/)), or raise an issue here on GitHub!
