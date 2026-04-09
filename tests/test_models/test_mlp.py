import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

import polars as pl
import yaml

import ringer_zero.models.mlp as mlp_module
from ringer_zero.datasets import ParquetDataset
from ringer_zero.models.mlp import MLPTrainingJob, add_inference
from ringer_zero.submitit import ExecutorConfig


def test_mlp_training_job(test_data_dir: Path, monkeypatch):
    with TemporaryDirectory() as output_dir:
        logging.info('Saving results to %s', output_dir)
        dataset_dir = test_data_dir / 'test_dataset'
        monkeypatch.setattr(mlp_module, 'get_n_folds',
                            lambda *args, **kwargs: 2)
        job = MLPTrainingJob(
            dataset_dir=dataset_dir,
            data_table='electron_ringer',
            rings_col='TrigEMClusterContainer.ringsE',
            kfold_table='standard_binning_kfold',
            label_col='label',
            fold_col='kfold',
            et_col='TrigEMClusterContainer.et',
            et_bins=(15000.0, 20000.0),
            eta_col='TrigEMClusterContainer.eta',
            eta_bins=(0.0, 0.8),
            output_dir=output_dir,
            tag='mlp',
            batch_size=32,
            inits=2,
            dry_run=True,
            executor_config=ExecutorConfig(
                cpus_per_task=1,
                executor_type='debug',
                logs_dir=Path('./logs'),
                name='test_mlp_training_job',
                slurm_array_parallelism=1,
                slurm_partition=None,
                stderr_to_stdout=True,
                timeout_min=5,
            ),
        )
        job.run()


def test_mlp_training_job_from_yaml(test_data_dir: Path, monkeypatch):
    with TemporaryDirectory() as output_dir:
        logging.info('Saving results to %s', output_dir)
        dataset_dir = test_data_dir / 'test_dataset'
        output_dir = Path(output_dir)

        cfg = {
            'data_table': 'data',
            'rings_col': 'trig_L2_calo_rings',
            'kfold_table': 'standard_binning_kfold',
            'label_col': 'label',
            'fold_col': 'kfold',
            'et_col': 'trig_L2_calo_et',
            'et_bins': [15000.0, 20000.0],
            'eta_col': 'trig_L2_calo_eta',
            'eta_bins': [0.0, 0.8],
            'tag': 'mlp',
            'batch_size': 32,
            'inits': 1,
            'dry_run': True,
            'executor_config': {
                'cpus_per_task': 1,
                'executor_type': 'debug',
                'logs_dir': './logs',
                'name': 'test_mlp_training_job_from_yaml',
                'slurm_array_parallelism': 1,
                'slurm_partition': None,
                'stderr_to_stdout': True,
                'timeout_min': 5,
            },
        }
        config_path = output_dir / 'mlp_training_job.yaml'
        with config_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f)

        calls: list[int] = []

        def fake_run_training(self, *args, **kwargs):
            calls.append(1)

        monkeypatch.setattr(MLPTrainingJob, 'run_training', fake_run_training)
        monkeypatch.setattr(mlp_module, 'get_n_folds',
                            lambda *args, **kwargs: 1)

        job = MLPTrainingJob.from_yaml(
            config_path,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
        )
        job.run()

        assert len(calls) == 1


class _FakeLoadedModel:
    def predict(self, data: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        data = data.select('id').with_columns(
            pl.lit(0.5, dtype=pl.Float32).alias('prediction')
        )
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        return data


def test_add_inference(test_data_dir: Path, monkeypatch):
    dataset_dir = test_data_dir / 'test_dataset'

    monkeypatch.setattr(
        MLPTrainingJob,
        'load_model',
        staticmethod(lambda *args, **kwargs: _FakeLoadedModel()),
    )

    add_inference(
        results_dir=test_data_dir / 'test_vqat_results',
        dataset_dir=dataset_dir,
        features_table='electron_ringer',
        inference_table='mlp_inference_results',
        eta_col='trig_L2_calo_eta',
        et_col='trig_L2_calo_et',
        rings_col='trig_L2_calo_rings',
    )

    dataset = ParquetDataset(dataset_dir=dataset_dir)
    table_path = dataset.get_table_path('mlp_inference_results')
    inference_df = pl.read_parquet(table_path)
    assert 'prediction' in inference_df.columns
    assert inference_df.height > 0
    shutil.rmtree(table_path)
