from pathlib import Path
import shutil
import polars as pl

from neuralnet.datasets import ParquetDataset
from neuralnet.models.vkan import VKANTrainingJob, add_inference


def test_vkan_pipeline_from_yaml(test_data_dir: Path):
    dataset_dir = test_data_dir / "test_dataset"
    training_dir = dataset_dir / "training" / "vkan"
    inference_dir = dataset_dir / "inference"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    if inference_dir.exists():
        shutil.rmtree(inference_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    job = VKANTrainingJob.from_yaml(
        test_data_dir / "vkan_training_job.yaml",
        output_dir=training_dir,
        dataset_dir=dataset_dir,
    )
    job.run()

    add_inference(
        results_dir=training_dir,
        dataset_dir=dataset_dir,
        features_table=job.data_table,
        inference_table="inference/vkan_inference_results",
    )

    dataset = ParquetDataset(dataset_dir=dataset_dir)
    inference_df = pl.read_parquet(
        dataset.get_table_path("inference/vkan_inference_results")
    )

    assert "output" in inference_df.columns
    assert len(inference_df) > 0
