"""
File:
    extract_results_plot.py

Authors:
    - Abir Riahi
    - Nicolas Raymond
    - Simon Giard-Leroux

Description:
    Parsing Python command line arguments to create results plots
"""

from src.visualization.VisualizationManager import VisualizationManager
from src.models.constants import SQUEEZE_NET_1_1, RESNET18
import argparse


def argument_parser():
    """
    This function defines a parser to enable user to retrieve results plot
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 extract_results_plots.py [model] [folder_prefix] '
                                           '[curve_label] [save_path]',
                                     description="This program enables user to get validation accuracy plots"
                                                 " from records.json files.")

    parser.add_argument('-m', '--model', type=str, default=SQUEEZE_NET_1_1,
                        choices=[SQUEEZE_NET_1_1, RESNET18],
                        help=f"Name of the model to train ({SQUEEZE_NET_1_1} or {RESNET18})")

    parser.add_argument('-fp', '--folder_prefix', type=str, default='generalization',
                        help=f"Start of the folders name from which to extract results")

    parser.add_argument('-c', '--curve_label', type=str, default='query_strategy',
                        help=f"Labels to use in order to compare validation accuracy curve")

    parser.add_argument('-s', '--save_path', type=str, default='accuracy_curve',
                        help=f"Name of the file containing the resulting plot")

    args = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("\n")

    return args


if __name__ == '__main__':

    # We extract arguments
    args = argument_parser()
    prefix = args.folder_prefix
    model = args.model
    curve_label = args.curve_label
    save_path = args.save_path

    # We produce the plot
    vm = VisualizationManager()
    vm.show_learning_curve(folder_prefix=prefix, model=model,
                           save_path=save_path, curve_label=curve_label)
