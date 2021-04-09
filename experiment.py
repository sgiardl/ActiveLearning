import argparse
import subprocess
from src.models.ActiveLearning import ActiveLearner


def argument_parser():
    """
    This function defines a parser to enable user to easily experiment different models
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]',
                                     description="This program enables user to train different "
                                                 "models of classification using passive or "
                                                 "active learning.")
    parser.add_argument('-m', '--model', type=str, default='SqueezeNet11',
                        choices=['SqueezeNet11', 'ResNet34'],
                        help='Name of the model to train '
                             '("ResNet34" or "SqueezeNet11")')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'EMNIST'],
                        help='Name of the dataset to learn on '
                             '("CIFAR10" or "EMNIST")')
    parser.add_argument('-ns', '--n_start', type=int, default=100,
                        help='The number of items that must be randomly '
                             'labeled in each class by the Expert')
    parser.add_argument('-nn', '--n_new', type=int, default=100,
                        help='The number of new items that must be labeled '
                             'within each active learning loop')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training epochs in each active learning '
                             'loop')
    parser.add_argument('-qs', '--query_strategy', type=str, default='least_confident',
                        choices=['random_sampling', 'least_confident', 'margin_sampling',
                                 'entropy_sampling'],
                        help='Query strategy of the expert')
    parser.add_argument('-en', '--experiment_name', type=str, default='test',
                        help='Name of the active learning experiment')
    parser.add_argument('-p', '--patience', type=int, default=4,
                        help='Maximal number of consecutive rounds without improvement')
    parser.add_argument('-b', '--batch_size', type=int, default=50,
                        help='Batch size of dataloaders storing train, valid and test set')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning rate of the model during training')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0,
                        help='Regularization term')
    parser.add_argument('-pt', '--pretrained', default=False, action='store_true',
                        help='Boolean indicating if the model used must be pretrained on '
                             'ImageNet')
    parser.add_argument('-da', '--data_aug', default=False, action='store_true',
                        help='Boolean indicating if we want data augmentation in the '
                             'training set')
    parser.add_argument('-nr', '--n_rounds', type=int, default=3,
                        help='Number of active learning rounds')

    args = parser.parse_args()

    print("The inputs are:")
    for arg in vars(args):
        print("{} is {}".format(arg, getattr(args, arg)))

    return args


def main():
    # Get argument_parser
    args = argument_parser()

    # Extract parameters values
    model = args.model
    dataset = args.dataset
    n_start = args.n_start
    n_new = args.n_new
    epochs = args.epochs
    query_strategy = args.query_strategy
    experiment_name = args.experiment_name
    patience = args.patience
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    pretrained = args.pretrained
    data_aug = args.data_aug
    n_rounds = args.n_rounds

    if pretrained:
        pretrained = True

    if data_aug:
        data_aug = True

    # Active learning model
    active_learner = ActiveLearner(model, dataset, n_start=n_start, n_new=n_new, epochs=epochs,
                                   query_strategy=query_strategy, experiment_name=experiment_name,
                                   patience=patience, batch_size=batch_size, lr=learning_rate,
                                   weight_decay=weight_decay, pretrained=pretrained, data_aug=data_aug)

    active_learner(n_rounds=n_rounds)


if __name__ == "__main__":
    main()
