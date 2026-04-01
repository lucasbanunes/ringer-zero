import os
import gc
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from .callbacks import EarlyStoppingCheckpoint


def training_torch(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sort: int,
    init: int,
    tag: str,
    model: nn.Module,
    output_dir: str,
    batch_size: int,
    et_bin: tuple[float, float],
    eta_bin: tuple[float, float],
    epochs: int = 5000,
    patience: int = 25,
    verbose: bool = True,
    dry_run: bool = False,
    **kwargs,
):
    output_dir = output_dir + "/tuned.%s.sort_%d.init_%d.model" % (tag, sort, init)
    if os.path.exists(output_dir):
        print("Output already exists")
        return {}
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    early_stopping = EarlyStoppingCheckpoint(patience=patience)
    max_epochs = 1 if dry_run else epochs

    start_time = datetime.now()
    model.train()

    for epoch in range(max_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss_tensor = criterion(val_outputs, y_val_tensor)
            val_loss = val_loss_tensor.item()

            val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
            val_correct = (val_predicted == y_val_tensor).sum().item()
            val_total = y_val_tensor.size(0)
            val_accuracy = val_correct / val_total if val_total > 0 else 0

        model.train()

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs}, "
                f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        should_stop = early_stopping.step(model=model, val_loss=val_loss)
        if should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    end_time = datetime.now()

    early_stopping.restore_best(model=model, device=device)

    torch.save(model.state_dict(), f"{output_dir}/model_weights.pth")

    history["patience"] = patience
    history["epochs"] = len(history["loss"])

    output_data = {
        "history": history,
        "model_type": "torch",
        "weights": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "metadata": {
            "et_bin": [str(et) for et in et_bin],
            "eta_bin": [str(eta) for eta in eta_bin],
            "sort": sort,
            "init": init,
            "tag": tag,
        },
        "time": (end_time - start_time),
    }

    with open(output_dir + "/results.pic", "wb") as f:
        pickle.dump(output_data, f)

    model = model.cpu()
    del train_loader, X_tensor, y_tensor, X_val_tensor, y_val_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history
