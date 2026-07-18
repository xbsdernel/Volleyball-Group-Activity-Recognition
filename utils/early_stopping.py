"""
Early stopping utility to halt training when validation loss stops
improving.
"""
from utils.checkpoint import save_checkpoint


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_val_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, avg_val_loss, model):
        """
        Call after each epoch with the current validation loss.
        Saves a checkpoint if the loss improved, otherwise increments
        the patience counter and flags should_stop when patience is
        exceeded.
        """
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.counter = 0
            save_checkpoint(model)
        else:
            self.counter += 1
            print(f"No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping!")
                self.should_stop = True

        return self.should_stop
