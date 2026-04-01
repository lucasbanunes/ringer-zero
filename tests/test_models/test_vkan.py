import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from ringer_zero.models.vkan import VKANTrainingJob
from ringer_zero.submitit import ExecutorConfig


def test_vkan_training_job(test_data_dir: Path):
    with TemporaryDirectory() as output_dir:
        logging.info(f"Saving results to {output_dir}")
        dataset_dir = test_data_dir / "test_dataset"
        output_dir = Path(output_dir)

        job = VKANTrainingJob(
            dataset_dir=dataset_dir,
            data_table="data",
            rings_col="trig_L2_calo_rings",
            kfold_table="standard_binning_kfold",
            label_col="label",
            fold_col="fold",
            et_col="trig_L2_calo_et",
            et_bins=(15000.0, 20000.0, 30000.0, 40000.0, 50000.0, float("inf")),
            eta_col="trig_L2_calo_eta",
            eta_bins=(0.0, 0.8, 1.37, 1.54, 2.37, 2.5),
            output_dir=output_dir,
            tag="vkan",
            grid_size=5,
            spline_order=3,
            batch_size=32,
            inits=1,
            epochs=5,
            patience=2,
            dry_run=True,
            executor_config=ExecutorConfig(
                cpus_per_task=1,
                executor_type="debug",
                logs_dir="./logs",
                name="test_vkan_training_job",
                slurm_array_parallelism=1,
                slurm_partition=None,
                stderr_to_stdout=True,
                timeout_min=5,
            ),
        )
        job.run()
        logging.info(f"Output files: {list(output_dir.glob('*'))}")


def test_vkan_training_job_from_yaml(test_data_dir: Path):
    with TemporaryDirectory() as output_dir:
        logging.info(f"Saving results to {output_dir}")
        dataset_dir = test_data_dir / "test_dataset"
        output_dir = Path(output_dir)

        job = VKANTrainingJob.from_yaml(
            test_data_dir / "vkan_training_job.yaml",
            output_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        job.run()
        logging.info(f"Output files: {list(output_dir.glob('*'))}")
