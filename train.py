import argparse
from src.models.ActiveLearning import ActiveLearner


def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]')
    parser.add_argument('--model', type=str, default='SqueezeNet11',
                        choices=['SqueezeNet11', 'ResNet34'])
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'EMNIST'])
    parser.add_argument('--n_start', type=int, default=100)
    parser.add_argument('--n_new', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--query_strategy', type=str, default='least_confident',
                        choices=['random_sampling', 'least_confident', 'margin_sampling',
                                 'entropy_sampling'])
    parser.add_argument('--experiment_name', type=str, default='least_confident',
                        choices=['random_sampling', 'least_confident', 'margin_sampling',
                                 'entropy_sampling'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

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

    active_learner = ActiveLearner(model, dataset, n_start=n_start, n_new=n_new, epochs=epochs,
                                   query_strategy=query_strategy, experiment_name=query_strategy,
                                   batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                   pretrained=False, data_aug=False)

    active_learner(n_rounds=3)
