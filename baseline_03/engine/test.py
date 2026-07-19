import torch
from sklearn.metrics import f1_score

from utils.metrics import compute_correct


def evaluate_test(model, test_loader, test_dataset, device):
    model.eval()

    total_acc_test = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)

            total_acc_test += compute_correct(outputs, labels)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    test_accuracy = (total_acc_test / len(test_dataset)) * 100

    test_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro"
    )

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")

    return test_accuracy, test_f1, true_labels, predicted_labels
