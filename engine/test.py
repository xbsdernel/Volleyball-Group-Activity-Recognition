import torch
from utils.metrics import compute_correct


def evaluate_test(model, test_loader, test_dataset, device):
    model.eval()
    with torch.no_grad():
        total_acc_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(inputs)
            acc = compute_correct(predictions, labels)
            total_acc_test += acc
    test_accuracy = (total_acc_test / len(test_dataset)) * 100
    print(f"Accuracy Score is: " f"{test_accuracy:.2f}%")
    return test_accuracy
