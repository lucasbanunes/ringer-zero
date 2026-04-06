import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve


def compute_sp(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute max SP and its (fa, pd) at the knee of the ROC curve."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # roc_curve requires both classes to be present.
    if np.unique(y_true).size < 2:
        return 0.0

    fa, pd, _ = roc_curve(y_true, y_score)
    sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
    knee = int(np.argmax(sp))
    return float(sp[knee])


class EarlyStoppingCheckpoint:
    def __init__(self, patience: int = 25, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = 0
        self.patience_counter = 0
        self.best_model_state = None

    def __call__(self, model: nn.Module, metric: float) -> bool:
        if metric > (self.best_metric + self.min_delta):
            self.best_metric = metric
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
