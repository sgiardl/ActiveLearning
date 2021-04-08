import matplotlib.pyplot as plt
from typing import Union
from src.models.Expert import Expert


class VisualizationManager:
    def __init__(self) -> None:
        """
        Visualization manager to generate charts.
        """
        self.legend_loc = 'upper right'
        self.marker = '.'
        self.train_label = 'Training'
        self.valid_label = 'Validation'

    def show_loss_acc_chart(self, results: dict,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Display and format chart of loss and accuracy per epoch

        :param results: Dictionary, contains each metric to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(results['Training Loss'], marker=self.marker, label=self.train_label)
        ax1.plot(results['Validation Loss'], marker=self.marker, label=self.valid_label)
        ax1.legend(loc=self.legend_loc)
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        ax2.plot(results['Training Accuracy'], marker=self.marker, label=self.train_label)
        ax2.plot(results['Validation Accuracy'], marker=self.marker, label=self.valid_label)
        ax2.legend(loc=self.legend_loc)
        ax2.set_ylim([0, 1])
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        fig.tight_layout()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    def show_labels_history(self, expert: Expert,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Plot the growth of labeled items per class throughout the active learning iteration

        :param expert: Expert class, contains the labels history to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """

        # We save the number of active learning iterations done
        x = range(len(expert.labeled_history[0]))
        for k, history in expert.labeled_history.items():
            plt.plot(x, history, marker=self.marker, label=expert.idx_to_class[k])

        # We set x-axis steps
        plt.xticks(x)

        # We set axis labels and legend
        plt.ylabel('Number of labeled images')
        plt.xlabel('Active learning iterations')
        plt.legend()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    def show_learning_curve(self, accuracy_dic: dict,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Method to show the learning curve, accuracy vs the number of active learning instance queries

        :param accuracy_dic: dictionary, keys are the query strategies and items are a list of accuracies by query
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """
        # Plot each accuracy list in accuracy dictionary with the corresponding query strategy
        for key in accuracy_dic:
            plt.plot(accuracy_dic[key], marker=self.marker, label=key)

        # We set axis labels and legend
        plt.ylabel('Training Accuracy')
        plt.xlabel('Number of Instance Queries')
        plt.legend(loc='lower right')
        plt.ylim([0, 1])

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()
