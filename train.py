import argparse
import sys
from src.models.ActiveLearning import ActiveLearner


def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]')
    parser.add_argument('--model', type=str, default='SqueezeNet11')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--n_start', type=int, default=100)
    parser.add_argument('--n_new', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--query_strategy', type=str, default='least_confident')
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    model = args.model
    dataset = args.dataset
    n_start = args.n_start
    n_new = args.n_new
    epochs = args.epochs
    query_strategy = args.query_strategy

    active_learner = ActiveLearner(model, dataset, n_start=n_start, n_new=n_new, epochs=epochs,
                                   query_strategy=query_strategy, experiment_name="test",
                                   batch_size=50, lr=0.0001, weight_decay=0,
                                   pretrained=False, data_aug=False)

    active_learner(n_rounds=3)
