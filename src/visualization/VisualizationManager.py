import matplotlib.pyplot as plt
from typing import Union
from src.models.TrainValidTestManager import TrainValidTestManager
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

    def show_loss_acc_chart(self, train_valid_test_manager: TrainValidTestManager,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Display and format chart of loss and accuracy per epoch

        :param train_valid_test_manager: TrainValidTestManager class,
                                         contains the losses and accuracies lists
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(train_valid_test_manager.train_loss_list, marker=self.marker, label=self.train_label)
        ax1.plot(train_valid_test_manager.valid_loss_list, marker=self.marker, label=self.valid_label)
        ax1.legend(self.legend_loc)
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        ax2.plot(train_valid_test_manager.train_accuracy_list, marker=self.marker, label=self.train_label)
        ax2.plot(train_valid_test_manager.valid_accuracy_list, marker=self.marker, label=self.valid_label)
        ax2.legend(self.legend_loc)
        ax2.set_ylim([0, 1])
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        fig.tight_layout()

        # We show the plot
        if show:
            plt.show()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

    def show_labels_history(self, expert: Expert,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Plot the growth of labeled items per class throughout the active learning iteration

        :param expert: Expert class, contains the labels history to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure
        """

        # We save the number of active learning iterations done
        x = range(len(expert.labeled_history[0]))
        for k, history in expert.labeled_history.items():
            plt.plot(x, history, label=expert.idx2class[k])

        # We set x-axis steps
        plt.xticks(x)

        # We set axis labels and legend
        plt.ylabel('Number of labeled images')
        plt.xlabel('Active learning iterations')
        plt.legend(self.legend_loc)

        # We show the plot
        if show:
            plt.show()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")
