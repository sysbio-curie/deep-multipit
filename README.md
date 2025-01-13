# deep-multipit

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository proposes end-to-end strategies for multimodal binary classification with attention weights. It gathers several 
PyTorch models as well as the scripts to reproduce the experiments from
[Vanguri *et al*, 2022](https://www.nature.com/articles/s43018-022-00416-8) and from our study:

[Captier, N., Lerousseau, M., Orlhac, F. et al. Integration of clinical, pathological, radiological, and transcriptomic data improves prediction for first-line immunotherapy outcome in metastatic non-small cell lung cancer. Nat Commun 16, 614 (2025).](https://doi.org/10.1038/s41467-025-55847-5)

* [Deep-multipit project](#deep-multipit)
    * [Installation](#installation)
    * [Multipit repository](#multipit-repository)
    * [Run scripts](#run-scripts)
      * [Using config files](#using-config-files)
      * [Run MSKCC scripts](#run-mskcc-scripts)
    * [Customization](#customization)
      * [Load your own data](#load-your-own-data)
      * [Customize multimodal models](#customize-multimodal-models)
      * [Customize training and testing](#customize-training-and-testing)
    * [Results folder architecture](#results-folder-architecture)
    * [Tensorboard visualization](#tensorboard-visualization)
    * [Acknowledgments](#acknowledgements)

  
## Installation
### Dependencies
- joblib (>= 1.2.0)
- lifelines (>= 0.27.4)
- numpy (>= 1.21.5)
- pandas (= 1.5.3)
- pyyaml (>= 6.0)
- PyTorch (>= 2.0.1)
- scikit-learn (>= 1.2.0)
- scikit-survival (>= 0.21.0)
- statsmodels (>= 0.13.5)
- Tensorboard (>= 2.8.0, optional)
- tqdm (>= 4.63.0)

### Install from source
Clone the repository:
```
git clone https://github.com/sysbio-curie/deep-multipit
```

## Multipit repository
This repository is paired with the [multipit](https://github.com/sysbio-curie/multipit) repository which provides a set of Python tools, compatible with scikit-learn,
to perform multimodal late and early fusion on tabular data as well as scripts to reproduce the experiments from our study.

## Run scripts
The code in this repository is run through Python scripts in [scripts](scripts) directory. You can either run existing scripts
(e.g. [train.py](scripts/train.py), [test.py](scripts/test.py), or [cross_validation.py](scripts/cross_validation.py)) or create 
your own scripts, using all the tools available in [dmultipit](dmultipit) package (you can use and update the auxiliary functions
available in [[_utils.py](scripts/_utils.py)]).

### Using config files
Scripts are run with command lines, with `.yaml` configuration files. Configuration are divided into two files: a config
file to define the multimodal model to use, located in [config_architecture](scripts/config_architecture) folder, and a
config file for all the settings of the experiment we want to run (e.g., training, test, or cross-validation), located in
[config_experiment](scripts/config_experiment) folder.

**Run train script**
```commandline
python train.py -c config_architecture/config_model.yaml -e config_experiment/config_train.yaml
```

You can also resume training from a specific checkpoint using:
```commandline
python train.py -r path_to_checkpoint -e config_experiment/config_train.yaml
```

**Run test script**
```commandline
python test.py -r path_to_checkpoint -e config_experiment/config_test.yaml
```

**Run cross-validation script**
````commandline
python cross_validation.py -c config_architecture/config_model.yaml -e config_experiment/config_cv.yaml
````
### Run MSKCC scripts
We provide all the code and data to implement the DyAM model and reproduce the experiments from [Vanguri *et al*, 2022](https://www.nature.com/articles/s43018-022-00416-8).

You first need to unzip the compressed folder [MSKCC.zip](data/MSKCC.zip) in the [data](data) directory. These raw data were extracted
from [synapse]( https://www.synapse.org/#!Synapse:syn26642505).

You can then run the scripts with the following command line:
```commandline
python cross_validation.py -c config_architecture\config_late_MSKCC.yaml -e config_experiment\config_cv_MSKCC.yaml
```
## Customization
The simplest and safest way to customize this project is by creating new Python scripts in the [scripts](scripts) directory 
and/or changing the settings in configurtion files.

For more complex customization you can also update the code in [dmultipit](dmultipit) package.

### Load your own data
**Define your own loading function**
If your data differ from the [MSKCC](data/MSKCC.zip) dataset or the TIPIT dataset described in our study you can create
your own loading function in [dmultipit/dataset/loader.py](dmultipit/dataset/loader.py) which should take as input the pathes
to the different unimodal data files and return the loaded data as well as the target to predict (see `load_TIPIT_multimoda`
or `load_MSKCC_multimoda`).   

**Define your own multimodal dataset**
You can also create your own Dataset in [dmultipit/dataset/dataset.py](dmultipit/dataset/dataset.py):
* Inheritating from `base.base_dataset.MultiModalDataset`
* Implementing abstract methods `fit_process` and `fit_multimodal_process`
* If needed, implementing new transformers for pre-processing in [dmultipit/dataset/transformers.py](dmultipit/dataset/transformers.py)
, inheritating from `base.base_transformers.UnimodalTransformer` or `base.base_transformers.MultimodalTransformer`

### Customize multimodal models
You can implement new multimodal models in [dmultipit/model/model.py](dmultipit/model/model.py), inheritating from `base.base_model.BaseModel`,
and using elements (potentialy new ones) from [dmultipit/model/attentions.py](dmultipit/model/attentions.py) and 
[dmultipit/model/embeddings.py](dmultipit/model/embeddings.py).

You can also implement our own loss or perfomance metrics in [dmultipit/model/loss.py](dmultipit/model/loss.py) and
[dmultipit/model/metric.py](dmultipit/model/metric.py) respectively.

### Customize training and testing 
`Trainer` and `Testing` classes can be updated in [dmultipit/trainer/trainer.py](dmultipit/trainer/trainer.py) and 
[dmultipit/testing/testing.py](dmultipit/testing/testing.py) respectively.

## Results folder architecture

By default, the results of the diffent experiments will be saved with the following folder architecture: 

```
  results_folder/
  │
  ├── train/ - results from train.py script
  │   ├── log/ - logdir for tensorboard and logging output
  │   │    └── model_name/
  │   │             └── run_id/ - training run id
  │   └── models/ - trained models are saved here
  │        └── model_name/
  │                 └── run_id/ - training run id
  │
  ├── test/ - results from test.py script
  │   ├── log/ - logdir for logging output
  │   │    └── model_name/
  │   │             └── run_id/ - training run id associated with the model to test
  │   │                     └── exp_id/ - testing experiment id
  │   └── saved/ - prediction results, embeddings, and attention weights are saved here
  │        └── model_name/
  │                 └── run_id/ - training run id associated with the model to test
  │                         └── exp_id/ - testing experiment id
  │
  └── cross-val/ - results from cross_validation.py script
      ├── log/ - logdir for tensorboard and logging output
      │    └── model_name/
      │             └── exp_id/ - cv experiment id
      └── models/ - trained models are saved here
      │    └── model_name/
      │             └── exp_id/ - cv experiment id
      └── saved/ - cross-validation results (i.e., predictions for the different folds and repeats) are saved here
           └── model_name/
                    └── exp_id/ - cv experiment id
```

* The name and location of the results_folder needs to be specified in the *config_experiment.yaml* file with the `save_dir` parameter.
* The model name needs to be specified in the *config_architecture.yaml* file with the `name` parameter.
* The training run id needs to be specified in the *config_experiment.yaml* file with the `run_ind` parameter (timestamp will be used if no id is specified).
* The experiment id needs to be specified in the *config_experiment.yaml* file with the `exp_id` parameter (timestamp will be used if no id is specified).   

**Note:** For modifying this default architecture you can update the initialization of the `ConfigParser` class in 
[dmultipit/parse_config.py](dmultipit/parse_config.py)

## Tensorboard visualization
Tensorboard visulaization is available with this project:
* Make sure to install Tensorboard first (either using `pip install tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX)).
* Make sure that tensorboard setting is turned on in your *config_experiment.yaml* file before running your training script (i.e., `tensorboard: true`).
* Run `tensorboard --logdir path_to_logdir` and server will open at `http://localhost:6006`.

**Note:** Tensorboard visualization is also available for cross-validation experiments, but we do not recommend to turn it on in case of multiple
repeats of the cv scheme (i.e., `n_repeats` > 1 in *config_experiment.yaml* file).

## Citing multipit

If you use deep-multipit in a scientific publication, we would appreciate citation to the [following paper](https://doi.org/10.1038/s41467-025-55847-5):

```
Captier, N., Lerousseau, M., Orlhac, F. et al. Integration of clinical, pathological, radiological, and transcriptomic data improves prediction for first-line immunotherapy outcome in metastatic non-small cell lung cancer. Nat Commun 16, 614 (2025). https://doi.org/10.1038/s41467-025-55847-5
```

## Acknowledgements

This repository was created as part of the PhD project of [Nicolas Captier](https://ncaptier.github.io/) in the
[Computational Systems Biology of Cancer group](https://institut-curie.org/team/barillot) and
the [Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.   

This repository was inspired by the [Pytorch Template Project](https://github.com/victoresque/pytorch-template) by Victor Huang.
