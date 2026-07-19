import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_metrics(total_loss_train_plot, total_loss_validation_plot,
                  total_acc_train_plot, total_acc_validation_plot,
                  total_f1_validation_plot, save_path=None):
    epochs = range(1, len(total_loss_train_plot) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, total_loss_train_plot, label="Training Loss")
    plt.plot(epochs, total_loss_validation_plot, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, total_acc_train_plot, label="Training Accuracy")
    plt.plot(epochs, total_acc_validation_plot, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, total_f1_validation_plot, label="Validation F1 Score")
    plt.title("Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, label_map, save_path=None):
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_map.keys(),
        yticklabels=label_map.keys()
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
