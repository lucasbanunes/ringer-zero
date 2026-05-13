import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl

from ringer_zero.datasets import ParquetDataset
from ringer_zero.models.mlp import MLPTrainingJob, add_inference


def test_mlp_pipeline_from_yaml(test_data_dir: Path):
    with TemporaryDirectory() as output_dir:
        logging.info(f"Saving results to {output_dir}")
        dataset_dir = test_data_dir / "test_dataset"
        output_dir = Path(output_dir)

        job = MLPTrainingJob.from_yaml(
            test_data_dir / "mlp_training_job.yaml",
            output_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        job.run()

        add_inference(
            results_dir=output_dir,
            dataset_dir=dataset_dir,
            features_table="electron_ringer",
            inference_table="mlp_inference_results",
        )

        dataset = ParquetDataset(dataset_dir=dataset_dir)
        inference_df = pl.read_parquet(dataset.get_table_path("mlp_inference_results"))

        assert "prediction" in inference_df.columns
        assert inference_df.height > 0
