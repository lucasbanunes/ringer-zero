import re
import gc
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import polars as pl
import torch

from ..datasets import ParquetDataset
from ..models.vkan import get_model, norm1


def load_model(
    model_path: Path,
    grid_size: int = 5,
    spline_order: int = 3,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    if model_path.is_dir():
        model_path = model_path / "model_weights.pth"

    device = torch.device(device)
    state_dict = torch.load(model_path, map_location=device)
    input_dim = state_dict["layers.0.base_weight"].shape[1]

    model = get_model(input_dim, grid_size, spline_order)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def parse_model_dir(
    model_dir: Path,
) -> tuple[tuple[float, float], tuple[float, float], int, int]:
    parts = model_dir.parts

    et_range = None
    eta_range = None
    fold = None
    init = None

    for part in parts:
        if et_range is None and part.startswith("et_"):
            left, right = part.removeprefix("et_").split("_", maxsplit=1)
            et_range = (float(left), float(right))
            continue

        if eta_range is None and part.startswith("eta_"):
            left, right = part.removeprefix("eta_").split("_", maxsplit=1)
            eta_range = (float(left), float(right))
            continue

        fold_init_match = re.fullmatch(r"fold_(\d+)_init_(\d+)", part)
        if fold_init_match is not None:
            fold = int(fold_init_match.group(1))
            init = int(fold_init_match.group(2))

    if et_range is None or eta_range is None or fold is None or init is None:
        raise ValueError(
            "Could not infer et/eta bins and fold from model path. "
            f"Expected path parts like et_* / eta_* / fold_*_init_*. Got: {model_dir}"
        )

    return et_range, eta_range, fold, init


def _discover_model_dirs(model_path: Path) -> list[Path]:
    if model_path.is_dir() and (model_path / "model_weights.pth").is_file():
        return [model_path]

    if model_path.is_dir():
        model_dirs = sorted(
            path for path in model_path.rglob("*.model") if path.is_dir()
        )
        if model_dirs:
            return model_dirs

    if model_path.suffix == ".model" and model_path.is_dir():
        return [model_path]

    raise ValueError(f"No model directories found under: {model_path}")


def _load_val_sp(model_dir: Path) -> float:
    result_file = model_dir / "results.pic"
    if not result_file.is_file():
        raise FileNotFoundError(f"Missing results file: {result_file}")

    with result_file.open("rb") as f:
        results = pd.read_pickle(f)

    return float(results["history"]["val_sp"][-1])


def _select_best_model_dirs(model_dirs: list[Path]) -> list[dict[str, object]]:
    best_by_region: dict[
        tuple[tuple[float, float], tuple[float, float]], dict[str, object]
    ] = {}

    for model_dir in model_dirs:
        et_bin, eta_bin, fold, init = parse_model_dir(model_dir)
        val_sp = _load_val_sp(model_dir)
        region_key = (et_bin, eta_bin)
        current_best = best_by_region.get(region_key)

        candidate = {
            "model_dir": model_dir,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
            "fold": fold,
            "init": init,
            "val_sp": val_sp,
        }

        if current_best is None or val_sp > current_best["val_sp"]:
            best_by_region[region_key] = candidate

    return [best_by_region[key] for key in sorted(best_by_region)]


def _get_ring_indexes() -> list[int]:
    ring_indexes = []
    ring_indexes += list(range(8 // 2))
    sum_rings = 8
    ring_indexes += list(range(sum_rings, sum_rings + (64 // 2)))
    sum_rings = 8 + 64
    ring_indexes += list(range(sum_rings, sum_rings + (8 // 2)))
    sum_rings = 8 + 64 + 8
    ring_indexes += list(range(sum_rings, sum_rings + (8 // 2)))
    sum_rings = 8 + 64 + 8 + 8
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    sum_rings = 8 + 64 + 8 + 8 + 4
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    ring_indexes += list(range(sum_rings, sum_rings + (4 // 2)))
    return ring_indexes


def _load_val_data_with_metadata(
    et_bin: tuple[float, float],
    eta_bin: tuple[float, float],
    dataset_dir: Path,
    data_table: str,
    kfold_table: str,
    et_col: str,
    eta_col: str,
    rings_col: str,
    label_col: str,
) -> tuple[torch.Tensor, pd.DataFrame]:
    dataset = ParquetDataset(dataset_dir=dataset_dir)
    ring_indexes = _get_ring_indexes()
    ring_cols = [f"rings_{i}" for i in ring_indexes]

    et = pl.col(et_col)
    eta = pl.col(eta_col).abs()

    data_df = (
        pl.scan_parquet(dataset.get_table_glob(data_table))
        .filter(
            et.is_between(et_bin[0], et_bin[1], closed="left")
            & eta.is_between(eta_bin[0], eta_bin[1], closed="left")
        )
        .select(
            "id",
            *[pl.col(rings_col).list.get(i).alias(f"rings_{i}") for i in ring_indexes],
        )
    )

    val_fold_df = pl.scan_parquet(dataset.get_table_glob(kfold_table)).select(
        "id",
        pl.col(label_col).cast(pl.Int32).alias("label"),
    )

    val_df = data_df.join(val_fold_df, on="id", how="inner").collect()
    val_rings = norm1(val_df.select(ring_cols).to_numpy().astype("float32"))
    val_X = torch.as_tensor(val_rings, dtype=torch.float32)

    metadata_df = val_df.select("id", "label").to_pandas()
    return val_X, metadata_df


def model_inference(
    model_path: str | Path,
    dataset_dir: Path,
    data_table: str,
    kfold_table: str,
    et_col: str,
    eta_col: str,
    rings_col: str,
    label_col: str,
    device: torch.device | str = "cpu",
    clear_cuda_cache: bool = True,
) -> pd.DataFrame:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available, but 'cuda' was specified. Please check your PyTorch installation and CUDA setup."
        )

    model_path = Path(model_path)
    model_dirs = _discover_model_dirs(model_path)
    best_model_dirs = _select_best_model_dirs(model_dirs)
    output_frames: list[pd.DataFrame] = []

    for model_info in tqdm(best_model_dirs):
        model_dir = model_info["model_dir"]
        et_bin = model_info["et_bin"]
        eta_bin = model_info["eta_bin"]
        model = load_model(model_path=model_dir, device=device)

        val_X, metadata_df = _load_val_data_with_metadata(
            et_bin=et_bin,
            eta_bin=eta_bin,
            dataset_dir=dataset_dir,
            data_table=data_table,
            kfold_table=kfold_table,
            et_col=et_col,
            eta_col=eta_col,
            rings_col=rings_col,
            label_col=label_col,
        )
        val_X = val_X.to(device)

        model.eval()
        with torch.inference_mode():
            logits = model(val_X)
            output = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            logits_np = logits.squeeze(1).detach().cpu().numpy()

        output_df = metadata_df.copy()
        output_df["output"] = output
        output_df["logits"] = logits_np

        output_frames.append(
            output_df[
                [
                    "id",
                    "output",
                    "logits",
                ]
            ]
        )

        del logits, output, logits_np, val_X, model, output_df, metadata_df

        if device.type == "cuda" and clear_cuda_cache:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return pd.concat(output_frames, ignore_index=True)
