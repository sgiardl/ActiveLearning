# Active Learning for Image Classification

This repository contains the term project for neural networks class (IFT780) 
at Universit√© de Sherbrooke (Winter 2021).

## Authors
* Abir Riahi (https://github.com/riaa3102)
* Nicolas Raymond (https://github.com/Rayn2402)
* Simon Giard-Leroux (https://github.com/sgiardl / https://github.com/gias2402)

## Introduction
This project implements active learning methods for deep neural networks. The goal is
to compare different active learning sampling criteria using different models and datasets
for the image classification task.

This figure shows the implemented pool-based active learning framework: 
![Image of pool-based active learning framework](/figures/PBAL.png?raw=true)

**Models:**
* ResNet34
* SqueezeNet 1.1

**Datasets:**
* EMNIST 'balanced' split (62 classes)
* CIFAR10 (10 classes)

**Query strategies:**
* Least Confidence
* Margin Sampling
* Entropy Sampling
* Random Sampling

## Installation
Install dependencies on a python environment
```
$ pip3 install -r requirements.txt
```

## Module Details: **experiment.py**

### Description

This program enables users to experiment different models of classification using passive or active learning.

### Arguments

| Short 	| Long              	    | Type    	| Default           	| Choices                                                                       	| Description                                                                   	|
|-------	|-----------------------	|---------	|-------------------	|-------------------------------------------------------------------------------	|-------------------------------------------------------------------------------	|
| `-m`    	| `--model`           	    | str     	| `'SqueezeNet11'`    	| [`'SqueezeNet11', 'ResNet34'`]                                                  	| Name of the model to train                                                    	|
| `-d`    	| `--dataset`         	    | str     	| `'CIFAR10'`         	| [`'CIFAR10', 'EMNIST'`]                                                         	| Name of the dataset to learn on                                               	|
| `-ns`   	| `--n_start`         	    | int     	| `100`               	|                                                                               	| The number of items that must be randomly labeled in each class by the Expert 	|
| `-nn`   	| `--n_new`           	    | int     	| `100`               	|                                                                               	| The number of new items that must be labeled within each active learning loop 	|
| `-e`    	| `--epochs`          	    | int     	| `50`                	|                                                                               	| Number of training epochs in each active learning loop                        	|
| `-qs`   	| `--query_strategy`  	    | str     	| `'least_confident'` 	| [`'random_sampling', 'least_confident', 'margin_sampling', 'entropy_sampling'`] 	| Query strategy of the expert                                                  	|
| `-en`   	| `--experiment_name` 	    | str     	| `'test'`            	|                                                                               	| Name of the active learning experiment                                        	|
| `-p`    	| `--patience`        	    | int     	| `4`                 	|                                                                               	| Maximal number of consecutive rounds without improvement                      	|
| `-b`    	| `--batch_size`      	    | int     	| `50`                	|                                                                               	| Batch size of dataloaders storing train, valid and test set                   	|
| `-lr`   	| `--learning_rate`   	    | float   	| `0.0001`            	|                                                                               	| Learning rate of the model during training                                    	|
| `-wd`   	| `--weight_decay`    	    | float   	| `0`                 	|                                                                               	| Regularization term                                                           	|
| `-pt`   	| `--pretrained`      	    | boolean 	| False             	|                                                                               	| Boolean indicating if the model used must be pretrained on ImageNet           	|
| `-da`   	| `--data_aug`        	    | boolean 	| False             	|                                                                               	| Boolean indicating if we want data augmentation in the training set           	|
| `-nr`   	| `--n_rounds`        	    | int     	| `3`                 	|                                                                               	| Number of active learning rounds                                              	|
| `-s`   	| `--init_sampling_seed`    | int     	| None                 	|                                                                               	| Seed value set when the expert labels items randomly in each class at start       |

``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To run a single experiment:
```
python3 experiment.py
python3 experiment.py --model='SqueezeNet11' --dataset='CIFAR10' --epochs=50
```

To run the generalization experiments batch:
```
python3 Experiments/Generalization.py
```

To run the pretraining experiments batch:
```
python3 Experiments/Pretraining.py
```

To run the query strategy experiments batch:
```
python3 Experiments/QueryStrategy.py
```

## Module Details: **extract_results_plot.py**

### Description

This program enables users to extract results and plot a learning curve for specific experiments that have been performed previously.

### Arguments

| Short 	| Long              	    | Type    	| Default           	| Choices                                                                       	| Description                                                                   	|
|-------	|-----------------------	|---------	|-------------------	|-------------------------------------------------------------------------------	|-------------------------------------------------------------------------------	|
| `-m`    	| `--model`           	    | str     	| `'SqueezeNet11'`    	| [`'SqueezeNet11', 'ResNet34'`]                                                  	| Name of the model to train                                                    	|
| `-fp`    	| `--folder_prefix`         	    | str     	| `'generalization'`         	|                                                          	| Start of the folders name from which to extract results                                               	|
| `-c`   	| `--curve_label`         	    | str     	| `query_strategy`               	|                                                                               	| Labels to use in order to compare validation accuracy curve
| `-s`   	| `--save_path`           	    | str     	| `accuracy_curve`               	|                                                                               	| Name of the file containing the resulting plot 	|

``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To plot the active learning curve for a particular experiments batch:
```
python3 extract_results_plots.py -m='SqueezeNet11' -fp='generalization' -c='query_strategy' -s='gen_accuracy'
```

## Project Organization

    ‚Ēú‚ĒÄ‚ĒÄ data
    ‚Ēā¬†¬† ‚ĒĒ‚ĒÄ‚ĒÄ raw            	 	<- The original, immutable data dump, where the data gets downloaded.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ Experiments        	 	<- Scripts to run the experiments as stated in the report.
    ‚Ēā   ‚ĒĒ‚ĒÄ‚ĒÄ Generalization.py    	<- Generalization experiments.
    ‚Ēā   ‚ĒĒ‚ĒÄ‚ĒÄ Helpers.py   	 	<- Helper functions.
    ‚Ēā   ‚ĒĒ‚ĒÄ‚ĒÄ Pretraining.py 	 	<- Pretraining experiments.
    ‚Ēā   ‚ĒĒ‚ĒÄ‚ĒÄ QueryStrategy.py     	<- Query strategy experiments.
    ‚Ēā
    ‚Ēā‚ĒÄ‚ĒÄ figures			<- Generated figures.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ models             	 	<- Trained and serialized models, model predictions, or model summaries
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ reports            	 	<- Generated analysis as PDF and LaTeX.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ src                	 	<- Source code for use in this project.
    ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ data           	 	<- Scripts to download or generate data
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ DataLoaderManager.py
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ DatasetManager.py
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚ĒĒ‚ĒÄ‚ĒÄ constants.py
    ‚Ēā   ‚Ēā
    ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ models         	 	<- Scripts to train models and then use trained models to make predictions
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ ActiveLearning.py
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ Expert.py
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚Ēú‚ĒÄ‚ĒÄ TrainValidTestManager.py
    ‚Ēā¬†¬† ‚Ēā¬†¬† ‚ĒĒ‚ĒÄ‚ĒÄ constants.py
    ‚Ēā   ‚Ēā
    ‚Ēā¬†¬† ‚ĒĒ‚ĒÄ‚ĒÄ visualization  	 	<- Scripts to create exploratory and results oriented visualizations
    ‚Ēā¬†¬†     ‚ĒĒ‚ĒÄ‚ĒÄ VisualizationManager.py
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ .gitignore			<- File that lists which files git can ignore.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ README.md          	 	<- The top-level README for developers using this project.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ experiment.py      	 	<- Argument parser to get command line arguments
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ extract_results_plots.py 	<- File to load results and plot active learning curves.
    ‚Ēā
    ‚Ēú‚ĒÄ‚ĒÄ requirements.txt   	 	<- The requirements file for reproducing the analysis environment,
    ‚Ēā					   generated with `pipreqs path/to/folder`
    ‚Ēā
    ‚ĒĒ‚ĒÄ‚ĒÄ test_environment.py      	<- Test environment to test the active learning loop.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>

