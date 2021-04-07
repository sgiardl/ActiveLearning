import matplotlib.pyplot as plt
from src.models.TrainValidTestManager import TrainValidTestManager

class VisualizationManager:
    def __init__(self):
        """

        """
        self.legend_loc = 'upper right'
        self.marker = '.'

    def show_loss_acc_chart(self, train_valid_test_manager: TrainValidTestManager):
        """

        :return:
        """
        # Display and format chart of loss and accuracy per epoch
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(train_valid_test_manager.train_loss_list, marker=self.marker, label='Training')
        ax1.plot(train_valid_test_manager.valid_loss_list, marker=self.marker, label='Validation')
        ax1.legend(self.legend_loc)
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        ax2.plot(train_valid_test_manager.train_accuracy_list, marker=self.marker, label='Training')
        ax2.plot(train_valid_test_manager.valid_accuracy_list, marker=self.marker, label='Validation')
        ax2.legend(self.legend_loc)
        ax2.set_ylim([0, 1])
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        fig.tight_layout()
        fig.show()
