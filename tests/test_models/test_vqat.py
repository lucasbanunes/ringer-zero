import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from ringer_zero.models.vqat import VQATTrainingJob
from ringer_zero.submitit import ExecutorConfig


def test_vqat_training_job(test_data_dir: Path):
    with TemporaryDirectory() as output_dir:
        logging.info(f'Saving results to {output_dir}')
        dataset_dir = test_data_dir / "test_dataset"
        output_dir = Path(output_dir)

        job = VQATTrainingJob(
            dataset_dir=dataset_dir,
            data_table='data',
            rings_col='trig_L2_calo_rings',
            kfold_table='standard_binning_kfold',
            label_col='label',
            fold_col='fold',
            et_col='trig_L2_calo_et',
            et_bins=(
                15000.0,
                20000.0,
                30000.0,
                40000.0,
                50000.0,
                float('inf')
            ),
            eta_col='trig_L2_calo_eta',
            eta_bins=(
                0.0,
                0.8,
                1.37,
                1.54,
                2.37,
                2.5
            ),
            output_dir=output_dir,
            tag='vqat',
            b0=22,
            i0=7,
            batch_size=32,
            inits=5,
            dry_run=True,
            executor_config=ExecutorConfig(
                cpus_per_task=1,
                executor_type='debug',
                logs_dir='./logs',
                name='test_vqat_training_job',
                slurm_array_parallelism=1,
                slurm_partition=None,
                stderr_to_stdout=True,
                timeout_min=5
            )
        )
        job.run()
        logging.info(f'Output files: {list(output_dir.glob("*"))}')


def test_vqat_training_job_from_yaml(test_data_dir: Path):
    with TemporaryDirectory() as output_dir:
        logging.info(f'Saving results to {output_dir}')
        dataset_dir = test_data_dir / "test_dataset"
        output_dir = Path(output_dir)

        job = VQATTrainingJob.from_yaml(
            test_data_dir / "vqat_training_job.yaml",
            output_dir=output_dir,
            dataset_dir=dataset_dir
        )
        job.run()
        logging.info(f'Output files: {list(output_dir.glob("*"))}')
