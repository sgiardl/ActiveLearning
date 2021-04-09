Active Learning for Image Classification
==============================

This repository contains the term project for neural networks class (IFT780) 
at Université de Sherbrooke (Winter 2021).

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

### **experiment.py**

**Description:**

This program enables user to experiment different models of classification using passive or active learning.

**Options:**

| Short 	| Long              	| Type    	| Default           	| Choices                                                                       	| Description                                                                   	|
|-------	|-------------------	|---------	|-------------------	|-------------------------------------------------------------------------------	|-------------------------------------------------------------------------------	|
| -m    	| --model           	| str     	| 'SqueezeNet11'    	| ['SqueezeNet11', 'ResNet34']                                                  	| Name of the model to train                                                    	|
| -d    	| --dataset         	| str     	| 'CIFAR10'         	| ['CIFAR10', 'EMNIST']                                                         	| Name of the dataset to learn on                                               	|
| -ns   	| --n_start         	| int     	| 100               	|                                                                               	| The number of items that must be randomly labeled in each class by the Expert 	|
| -nn   	| --n_new           	| int     	| 100               	|                                                                               	| The number of new items that must be labeled within each active learning loop 	|
| -e    	| --epochs          	| int     	| 50                	|                                                                               	| Number of training epochs in each active learning loop                        	|
| -qs   	| --query_strategy  	| str     	| 'least_confident' 	| ['random_sampling', 'least_confident', 'margin_sampling', 'entropy_sampling'] 	| Query strategy of the expert                                                  	|
| -en   	| --experiment_name 	| str     	| 'test'            	|                                                                               	| Name of the active learning experiment                                        	|
| -p    	| --patience        	| int     	| 4                 	|                                                                               	| Maximal number of consecutive rounds without improvement                      	|
| -b    	| --batch_size      	| int     	| 50                	|                                                                               	| Batch size of dataloaders storing train, valid and test set                   	|
| -lr   	| --learning_rate   	| float   	| 0.0001            	|                                                                               	| Learning rate of the model during training                                    	|
| -wd   	| --weight_decay    	| float   	| 0                 	|                                                                               	| Regularization term                                                           	|
| -pt   	| --pretrained      	| boolean 	| False             	|                                                                               	| Boolean indicating if the model used must be pretrained on ImageNet           	|
| -da   	| --data_aug        	| boolean 	| False             	|                                                                               	| Boolean indicating if we want data augmentation in the training set           	|
| -nr   	| --n_rounds        	| int     	| 3                 	|                                                                               	| Number of active learning rounds                                              	|

**Examples of basic use:**

```
python3 experiment.py
python3 experiment.py --model='SqueezeNet11' --dataset='CIFAR10' --epochs=50
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
