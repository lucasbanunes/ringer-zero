import torch
import torch.nn as nn


class EarlyStoppingCheckpoint:
    def __init__(self, patience: int = 25, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def step(self, model: nn.Module, val_loss: float) -> bool:
        improved = val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def restore_best(self, model: nn.Module, device: torch.device):
        if self.best_model_state is None:
            return
        model.load_state_dict(
            {key: value.to(device) for key, value in self.best_model_state.items()}
        )
