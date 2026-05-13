from pathlib import Path
import shutil
import polars as pl

from neuralnet.datasets import ParquetDataset
from neuralnet.models.mlp import MLPTrainingJob, add_inference


def test_mlp_pipeline_from_yaml(test_data_dir: Path):
    dataset_dir = test_data_dir / "test_dataset"
    training_dir = dataset_dir / "training" / "mlp"
    inference_dir = dataset_dir / "inference"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    if inference_dir.exists():
        shutil.rmtree(inference_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    job = MLPTrainingJob.from_yaml(
        test_data_dir / "mlp_training_job.yaml",
        output_dir=training_dir,
        dataset_dir=dataset_dir,
    )
    job.run()

    add_inference(
        results_dir=training_dir,
        dataset_dir=dataset_dir,
        features_table="electron_ringer",
        inference_table="inference/mlp_inference_results",
    )

    dataset = ParquetDataset(dataset_dir=dataset_dir)
    inference_df = pl.read_parquet(
        dataset.get_table_path("inference/mlp_inference_results")
    )

    assert "prediction" in inference_df.columns
    assert inference_df.height > 0
