import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from utils.metrics import compute_correct


def validate_one_epoch(model, val_loader, criterion, device, epoch, epochs):
    true_labels = []
    predicted_labels = []
    model.eval()
    total_loss_val = total_acc_val = 0
    seen_val = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False)
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            batch_correct = compute_correct(outputs, labels)
            total_acc_val += batch_correct
            seen_val += labels.size(0)

            val_bar.set_postfix(
                loss=f"{val_loss.item():.4f}",
                acc=f"{total_acc_val / seen_val * 100:.2f}%",
            )

    avg_val_loss = total_loss_val / len(val_loader)
    val_accuracy = total_acc_val / seen_val * 100
    val_f1 = f1_score(true_labels, predicted_labels, average="macro")

    return avg_val_loss, val_accuracy, val_f1
