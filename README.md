# nnp-pre-training


Code, data and model weights for the pre-print:

<div align="center">

> **[Synthetic pre-training for neural-networkinteratomic potentials]()**\
> _[John Gardner](https://jla-gardner.github.io), [Kathryn Baker]() and [Volker Deringer](http://deringer.chem.ox.ac.uk)_

</div>

---

## Repo Overview

-  **[src/](src/)** contains the source code for performing synthetic pre-training.
-  **[scripts/](scripts/)** contains the scripts used to run the experiments presented in the paper.
-  **[notebooks/](./plotting/analysis.ipynb)** contains the notebooks to generate the plots in the paper.
-  **[data/labels](data/labels)** contains the synthetic labels generated as part of this work.
-  **[figures/](figures/)** contains the figures used in the paper.
-  **[results/](results/)** contains the results of the experiments presented in the paper. See notebooks for how to interpret these.
-  **[models/](models/)** contains the model weights for the pre-trained models used in the paper. See below for how to load these. 

---

## Reproducing our results

### 1. Clone the repo

```bash
git clone https://github.com/jla-gardner/nnp-pre-training.git
cd nn-pre-training
```

### 2. Install dependencies

We strongly recommend using a virtual environment. With `conda` installed, this is as simple as:

```bash
conda create -n synthetic python=3.8 -y
conda activate synthetic
```

All dependencies can then be installed with:

```bash
pip install -r requirements.txt
```

### 3. Run the experiments

The scripts for running the experiments are in `./scripts/`. To run one of these, do:

```bash
./run <script-name> <keyword-options>
```

e.g. `./run direct_training dataset_name=C-GAP-17 labels=dft num_layers=2`

--- 

## Loading the pre-trained models

Each synthetically pre-trained model has been provided as a `.pth` file. These can be loaded using the NequIP library by making a model with the following configuration:

```yaml
default_dtype: float32
nonlinearity_type: gate
BesselBasis_trainable: true
parity: true
r_max: 4.0
num_layers: 4
num_features: 32
l_max: 1

model_builders: 
    - SimpleIrrepsConfig
    - EnergyModel
    - PerSpeciesRescale
    - ForceOutput
    - RescaleEnergyEtc 
    - initialize_from_state

per_species_rescale_shifts: 0.0
per_species_rescale_scales: 1.0
global_rescale_shift: null
global_rescale_scale: null

initial_model_state: <path-to-model>
```

The models are named according to the following convention:
`<synthetic-label-source>-<number-of-pre-training-structures>`
