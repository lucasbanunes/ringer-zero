import yaml
import polars as pl
from pathlib import Path
from keras import Sequential, Input
from keras.layers import Dense

from neuralnet.models.mlp import (
    add_uniform_ptq_inference,
    MLPUniformPTQInference,
    BinnedKerasExpertCommittee,
)


def test_add_uniform_ptq_inference_runs(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    results_dir = tmp_path / "results"
    dataset_dir.mkdir()
    results_dir.mkdir()

    config = {
        "dataset_dir": str(dataset_dir),
        "results_dir": str(results_dir),
        "features_table": "features",
        "inference_table": "inference",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # Minimal results/best_models DataFrames (will be written but content irrelevant)
    results_df = pl.DataFrame({"a": [1]})
    best_models_df = pl.DataFrame({"a": [1]})

    # Simple Keras model that accepts two features
    keras_model = Sequential([Input(shape=(2,)), Dense(1, activation="sigmoid")])

    # Build a BinnedKerasExpertCommittee-like object via dict coercion
    model_dict = dict(
        bins=[dict(col=pl.col("eta").abs(), lower=0.0, upper=10.0, closed="left")],
        features=[pl.col("f0"), pl.col("f1")],
        model=keras_model,
    )
    loaded_model = BinnedKerasExpertCommittee(models=[model_dict])

    # Patch MLPTrainingJob.load_model to return our simple objects
    def fake_load_model(results_dir_arg, eta_col, et_col, rings_col):
        return results_df, best_models_df, loaded_model

    monkeypatch.setattr(
        "neuralnet.models.mlp.MLPTrainingJob.load_model", fake_load_model
    )

    # Patch ParquetDataset used in the module so paths are predictable
    class DummyParquetDataset:
        def __init__(self, dataset_dir=None, **kwargs):
            self.dataset_dir = dataset_dir

        def get_table_glob(self, table):
            return "ignored"

        def get_table_path(self, table):
            return tmp_path / "out" / table

    monkeypatch.setattr("neuralnet.models.mlp.ParquetDataset", DummyParquetDataset)

    # Make pl.scan_parquet return a lazy frame with the expected columns
    df = pl.DataFrame({"id": [1, 2], "f0": [0.1, 0.2], "f1": [0.3, 0.4], "eta": [0.1, 0.2]})
    monkeypatch.setattr(pl, "scan_parquet", lambda path: df.lazy())

    # Call the CLI wrapper by passing the config file path (string),
    # which `from_yaml` now accepts as a path.
    job = add_uniform_ptq_inference(str(config_path))

    assert isinstance(job, MLPUniformPTQInference)
