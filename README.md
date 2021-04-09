Active Learning for Image Classification
==============================

This repository contains the term project for neural networks class (IFT780) 
at the University of Sherbrooke (Winter 2021).

## Introduction
This project implements active learning methods for deep neural networks. The goal is
to compare different active learning sampling criteria using different models and datasets
for the image classification task.

This figure shows the implemented pool-based active learning framework: 
![Image of pool-based active learning framework](/report/figures/PBAL.png?raw=true)

**Models:**
* ResNet34
* SqueezeNet 1.1

**Datasets:**
* EMNIST (62 classes)
* CIFAR10 (10 classes)

**Query strategies:**
* Least Confidence (LC)
* Margin Sampling
* Entropy Sampling
* Random Sampling

## Installation
Install dependencies on a python environment
```
$ pip3 install -r requirements.txt
```

## Module Details

### **train.py**

**Description:**

This program enables user to train different models of classification using passive or active learning.

**Options:**

* --model: Name of the model to train
* --dataset: Name of the dataset to learn on
* --n_start: Number of items that must be randomly labeled in each class by the Expert
* --n_new: Number of new items that must be labeled within each active learning loop
* --epochs: Number of training epochs in each active learning loop
* --query_strategy: Query strategy of the expert
* --experiment_name: Name of the active learning experiment
* --batch_size: Batch size of dataloaders storing train, valid and test set
* --lr: Learning rate of the model during training
* --weight_decay: The regularization term
* --pretrained
* --data_aug
* --n_rounds: Number of active learning rounds

**Examples of basic use:**

```
python3 train.py
python3 train.py --model='SqueezeNet11' --dataset='CIFAR10' --epochs=50
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Authors
* Abir Riahi
* Nicolas Raymond
* Simon Giard-Leroux
