import gc
from collections.abc import Callable, Mapping
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

ModelInfo = Mapping[str, object]
SelectModelsFn = Callable[[Path], list[ModelInfo]]
InferModelFn = Callable[[ModelInfo, torch.device], pd.DataFrame]


def model_inference(
    model_path: str | Path,
    select_models: SelectModelsFn,
    infer_model: InferModelFn,
    device: torch.device | str = "cpu",
    clear_cuda_cache: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available, but 'cuda' was specified. Please check your PyTorch installation and CUDA setup."
        )

    model_path = Path(model_path)
    selected_models = select_models(model_path)
    output_frames: list[pd.DataFrame] = []

    iterator = tqdm(selected_models, disable=not show_progress)
    for model_info in iterator:
        output_frames.append(infer_model(model_info, device))

        if device.type == "cuda" and clear_cuda_cache:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if not output_frames:
        raise ValueError(f"No models selected for inference from: {model_path}")

    return pd.concat(output_frames, ignore_index=True)
