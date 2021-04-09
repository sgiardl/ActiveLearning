import argparse
import sys
from src.models.ActiveLearning import ActiveLearner

REQUIRED_PYTHON = "python3"


def argument_parser():
    """
    This function defines a parser to enable user to easily experiment different models
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]',
                                     description="This program enables user to train different "
                                                 "models of classification using passive or "
                                                 "active learning.")
    parser.add_argument('--model', type=str, default='SqueezeNet11',
                        choices=['SqueezeNet11', 'ResNet34'],
                        help='Name of the model to train '
                             '("ResNet34" or "SqueezeNet11")')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'EMNIST'],
                        help='Name of the dataset to learn on '
                             '("CIFAR10" or "EMNIST")')
    parser.add_argument('--n_start', type=int, default=100,
                        help='The number of items that must be randomly '
                             'labeled in each class by the Expert')
    parser.add_argument('--n_new', type=int, default=100,
                        help='The number of new items that must be labeled '
                             'within each active learning loop')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs in each active learning '
                             'loop')
    parser.add_argument('--query_strategy', type=str, default='least_confident',
                        choices=['random_sampling', 'least_confident', 'margin_sampling',
                                 'entropy_sampling'],
                        help='Query strategy of the expert')
    parser.add_argument('--experiment_name', type=str, default='exp',
                        help='Name of the active learning experiment')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size of dataloaders storing train, valid and test set')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate of the model during training')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='The regularization term')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='Bool indicating if the model used must be pretrained on '
                             'ImageNet')
    parser.add_argument('--data_aug', default=False, action='store_true',
                        help='Bool indicating if we want data augmentation in the '
                             'training set')
    parser.add_argument('--n_rounds', type=int, default=3,
                        help='Number of active learning rounds')
    parser.add_argument('--patience', type=int, default=4,
                        help='Maximal number of consecutive rounds without accuracy improvement')
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    # Extract parameters values
    model = args.model
    dataset = args.dataset
    n_start = args.n_start
    n_new = args.n_new
    epochs = args.epochs
    query_strategy = args.query_strategy
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    pretrained = args.pretrained
    data_aug = args.data_aug
    n_rounds = args.n_rounds
    patience = args.patience

    if pretrained:
        pretrained = True

    if data_aug:
        data_aug = True

    # Active learning model
    active_learner = ActiveLearner(model, dataset, n_start=n_start, n_new=n_new, epochs=epochs,
                                   query_strategy=query_strategy, experiment_name=query_strategy,
                                   batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                   pretrained=pretrained, data_aug=data_aug, patience=patience)

    active_learner(n_rounds=n_rounds)
