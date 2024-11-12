# deep-multipit

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

This repository proposes end-to-end strategies for multimodal binary classification with attention weights. It gathers several 
PyTorch models as well as the scripts to reproduce the experiments from
[Vanguri *et al*, 2022](https://www.nature.com/articles/s43018-022-00416-8) and from our study:

"Integration of clinical, pathological, radiological, and transcriptomic data improves the prediction of first-line immunotherapy outcome in metastatic non-small cell lung cancer"

**Preprint:** https://doi.org/10.1101/2024.06.27.24309583


* [Deep-multipit project](#deep-multipit)
    * [Installation](#installation)
    * [Run scripts](#run-scripts)
      * [Using config files](#using-config-files)
      * [Run MSKCC scripts](#run-mskcc-scripts)
    * [Customization](#customization)
      * [Load your own data](#load-your-own-data)
      * [Customize multimodal models](#customize-multimodal-models)
      * [Customize training and testing](#customize-training-and-testing)
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

## Run scripts

### Using config files
### Run MSKCC scripts

## Customization

### Load your own data

### Customize multimodal models

### Customize training and testing 

## Tensorboard visualization

## Acknowledgements

This repository was created as part of the PhD project of Nicolas Captier in the
[Computational Systems Biologie of Cancer group](https://institut-curie.org/team/barillot) and
the [ Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.   

This repository was inspired by the [Pytorch Template Project](https://github.com/victoresque/pytorch-template) by Victor Huang.