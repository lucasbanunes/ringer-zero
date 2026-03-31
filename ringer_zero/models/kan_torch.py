import torch
import torch.nn as nn
from efficient_kan import KAN


def get_torch_kan_model(input_dim: int, grid_size: int, spline_order: int) -> nn.Module:
    model = KAN([input_dim, 5, 1], grid_size=grid_size, spline_order=spline_order)
    return model
