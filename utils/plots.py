import os
import matplotlib.pyplot as plt


def plot_metrics(total_loss_train_plot, total_loss_validation_plot,
                  total_acc_train_plot, total_acc_validation_plot,
                  save_path=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(total_loss_train_plot, label='Training Loss')
    axs[0].plot(total_loss_validation_plot, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(total_acc_train_plot, label='Training Accuracy')
    axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
