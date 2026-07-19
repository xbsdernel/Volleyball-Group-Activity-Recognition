from tqdm.auto import tqdm
from utils.metrics import compute_correct


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_acc_train = total_loss_train = 0
    seen_train = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
    for inputs, labels in train_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()
        optimizer.step()

        batch_correct = compute_correct(outputs, labels)
        total_acc_train += batch_correct
        seen_train += labels.size(0)

        train_bar.set_postfix(
            loss=f"{train_loss.item():.4f}",
            acc=f"{total_acc_train / seen_train * 100:.2f}%",
        )

    avg_train_loss = total_loss_train / len(train_loader)
    train_accuracy = total_acc_train / seen_train * 100

    return avg_train_loss, train_accuracy
