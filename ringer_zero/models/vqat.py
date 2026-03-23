#!/usr/bin/env python
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Annotated
import pandas as pd
import numpy as np
from keras import Sequential
import hgq.layers as qlayers
from hgq.config import QuantizerConfigScope
from hgq.constraints import Constant
from pydantic import BaseModel, Field
import typer
import duckdb
import yaml

from ringer_zero.tunning import training, RefType
from ringer_zero import get_logger
from ringer_zero.datasets import ParquetDataset
from ..slurm import SlurmConfig


def get_model(b0: int, i0: int) -> Sequential:
    with (
        QuantizerConfigScope(
            place={'weight', 'bias'},
            q_type='kbi',
            k0=True,
            b0=b0,
            i0=i0,
            bc=Constant(b0),
            ic=Constant(i0)
        )
    ):
        model = Sequential([
            qlayers.QDense(5, activation='relu',
                           input_shape=(50,), iq_conf=None),
            qlayers.QDense(1, activation='sigmoid'),
        ])

    return model


def quantizer(rings):

    rings_q15 = np.round(rings * 32768)
    rings_q15 = np.clip(rings_q15, -32768, 32767).astype(np.int16)
    rings_q15 = rings_q15.astype(np.float32) / 32768.0

    return rings_q15


def get_data_query(
        rings_col: str,
        rings_indexes: list[int],
        label_col: str,
        fold_col: str,
        fold: int,
        fold_signal: str,
        et_col: str,
        et_bin_left: float,
        et_bin_right: float,
        eta_col: str,
        eta_bin_left: float,
        eta_bin_right: float,
        data_table_glob: str,
        kfold_table_glob: str) -> str:
    scalar_rings_indexes = ',\n    '.join(
        [f"data.{rings_col}[{i+1}] as rings_{i}" for i in rings_indexes])
    return f"""
SELECT
    {scalar_rings_indexes},
    CAST(kfold.{label_col} AS UTINYINT) as label
FROM read_parquet('{data_table_glob}') as data
LEFT JOIN read_parquet('{kfold_table_glob}') as kfold
ON data.id = kfold.id
WHERE data.{et_col} >= {et_bin_left} AND
      data.{et_col} < {et_bin_right} AND
      abs(data.{eta_col}) >= {eta_bin_left} AND
      abs(data.{eta_col}) < {eta_bin_right} AND
      kfold.{fold_col} {fold_signal} {fold} AND
      kfold.{label_col} IS NOT NULL AND
      kfold.{fold_col} IS NOT NULL;
"""


def get_n_folds(kfold_table_glob: str, fold_col: str) -> int:
    with duckdb.connect(':memory:') as conn:
        query = f"""
        SELECT MAX({fold_col}) + 1 as n_folds
        FROM read_parquet('{kfold_table_glob}')
        WHERE {fold_col} IS NOT NULL;
        """
        return conn.execute(query).fetchone()[0]


def norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data/norms[:, None]


class VQATTrainingJob(BaseModel):
    dataset_dir: Annotated[
        Path,
        Field(
            description='Directory containing the parquet dataset'
        )]
    data_table: Annotated[
        str,
        Field(
            description='Name of the data table in the parquet dataset'
        )]
    rings_col: Annotated[
        str,
        Field(
            description='Name of the rings column in the data table'
        )]
    kfold_table: Annotated[
        str,
        Field(
            description='Name of the kfold table in the parquet dataset'
        )]
    label_col: Annotated[
        str,
        Field(
            description='Name of the label column in the kfold table'
        )]
    fold_col: Annotated[
        str,
        Field(
            description='Name of the fold column in the kfold table'
        )]
    et_col: Annotated[
        str,
        Field(
            description='Name of the et column in the data table'
        )]
    et_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='et bin to select from the dataset. Should be in the format "et_bin_left,et_bin_right". Example: "20000,40000"'
        )]
    eta_col: Annotated[
        str,
        Field(
            description='Name of the eta column in the data table'
        )]
    eta_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='eta bin to select from the dataset. Should be in the format "eta_bin_left,eta_bin_right". Example: "0.0,0.8"'
        )]
    output_dir: Annotated[
        Path | None,
        Field(
            description='Directory to save the training results'
        )] = None
    tag: Annotated[
        str,
        Field(
            description='Tag to identify the training run'
        )]
    b0: Annotated[
        int,
        Field(
            description='Number of bits for weights and biases quantization'
        )]
    i0: Annotated[
        int,
        Field(
            description='Number of integer bits for weights and biases quantization'
        )]
    batch_size: Annotated[
        int,
        Field(
            description='Batch size for training'
        )] = 1024
    inits: Annotated[
        int,
        Field(
            description='Number of initializations'
        )] = 5
    dry_run: Annotated[
        bool,
        Field(
            description='Perform a dry run without actually training'
        )] = False
    slurm_config: Annotated[
        SlurmConfig | None,
        Field(
            description='Slurm configuration for running the training job on a Slurm cluster'
        )] = None

    def model_post_init(self, context):
        if isinstance(self.et_bins, str):
            self.et_bins = tuple(float(x) for x in self.et_bins.split(','))
        if isinstance(self.eta_bins, str):
            self.eta_bins = tuple(float(x) for x in self.eta_bins.split(','))

        if self.output_dir is None:
            self.output_dir = Path.cwd() / 'vqat_training_job'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return super().model_post_init(context)

    @classmethod
    def from_yaml(cls, yaml_file: Path, **kwargs) -> 'VQATTrainingJob':
        """Load VQATTrainingJob from a YAML file."""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        for key, value in kwargs.items():
            data[key] = value
        return cls(**data)

    def load_data(self,
                  fold: int,
                  et_bin_left: float,
                  et_bin_right: float,
                  eta_bin_left: float,
                  eta_bin_right: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and validation data for the given fold."""
        # We select 1/2 of rings in each layer
        # pre-sample - 8 rings
        # EM1 - 64 rings
        # EM2 - 8 rings
        # EM3 - 8 rings
        # Had1 - 4 rings
        # Had2 - 4 rings
        # Had3 - 4 rings
        rings_indexes = []
        # rings presmaple
        rings_indexes += list(range(8//2))

        # EM1 list
        sum_rings = 8
        rings_indexes += list(range(sum_rings, sum_rings+(64//2)))

        # EM2 list
        sum_rings = 8+64
        rings_indexes += list(range(sum_rings, sum_rings+(8//2)))

        # EM3 list
        sum_rings = 8+64+8
        rings_indexes += list(range(sum_rings, sum_rings+(8//2)))

        # HAD1 list
        sum_rings = 8+64+8+8
        rings_indexes += list(range(sum_rings, sum_rings+(4//2)))

        # HAD2 list
        sum_rings = 8+64+8+8+4
        rings_indexes += list(range(sum_rings, sum_rings+(4//2)))

        # HAD3 list
        sum_rings = 8+64+8+8+4+4
        rings_indexes += list(range(sum_rings, sum_rings+(4//2)))

        dataset = ParquetDataset(dataset_dir=self.dataset_dir)

        train_query = get_data_query(
            rings_col=self.rings_col,
            rings_indexes=rings_indexes,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=fold,
            fold_signal='!=',
            et_col=self.et_col,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_col=self.eta_col,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
            data_table_glob=dataset.get_table_glob(self.data_table),
            kfold_table_glob=dataset.get_table_glob(self.kfold_table)
        )

        val_query = get_data_query(
            rings_col=self.rings_col,
            rings_indexes=rings_indexes,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=fold,
            fold_signal='=',
            et_col=self.et_col,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_col=self.eta_col,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
            data_table_glob=dataset.get_table_glob(self.data_table),
            kfold_table_glob=dataset.get_table_glob(self.kfold_table)
        )

        with duckdb.connect(':memory:') as conn:
            train_df = conn.execute(train_query) \
                .fetch_arrow_table().to_pandas(types_mapper=pd.ArrowDtype)
            val_df = conn.execute(val_query) \
                .fetch_arrow_table().to_pandas(types_mapper=pd.ArrowDtype)

        # The dataframes only have rings and the label,
        # we need to separate them and convert the rings to normalized and quantized numpy arrays
        val_label = val_df.pop('label').values
        val_rings = norm1(quantizer(val_df.values.astype(np.float32)))
        del val_df  # Frees memory premptively
        train_label = train_df.pop('label').values
        train_rings = norm1(quantizer(train_df.values.astype(np.float32)))
        del train_df

        return train_rings, val_rings, train_label.astype(np.int32), val_label.astype(np.int32)

    def load_ref(self,
                 et_bin_left: float,
                 et_bin_right: float,
                 eta_bin_left: float,
                 eta_bin_right: float
                 ) -> RefType:
        dataset = ParquetDataset(dataset_dir=self.dataset_dir)
        ref_query = f"""
        SELECT *
        FROM read_parquet('{str(dataset.get_table_glob("ref"))}') as ref
        WHERE ref.et_bin_lower = {et_bin_left} AND
              ref.et_bin_upper = {et_bin_right} AND
              ref.eta_bin_lower = {eta_bin_left} AND
              ref.eta_bin_upper = {eta_bin_right};
        """
        ref = defaultdict(lambda: defaultdict(dict))
        with duckdb.connect(':memory:') as conn:
            ref_df = conn.execute(ref_query).fetch_df()

        for _, row in ref_df.iterrows():
            ref[row['criteria']][row['sample_type']
                                 ][row['total_or_passed']] = row['value']
        for key in ref:
            ref[key] = dict(ref[key])
        ref = dict(ref)

        return ref

    def run_training(self,
                     et_bin_left: float,
                     et_bin_right: float,
                     eta_bin_left: float,
                     eta_bin_right: float,
                     fold: int,
                     init: int,):

        logger = get_logger()
        logger.info(
            f'Loading data for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}')
        ref = self.load_ref(
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right
        )
        X, X_val, y, y_val = self.load_data(
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right
        )
        output_dir = self.output_dir / f'et_{et_bin_left}_{et_bin_right}' / \
            f'eta_{eta_bin_left}_{eta_bin_right}' / f'fold_{fold}_init_{init}'
        output_dir.mkdir(parents=True, exist_ok=True)
        training(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            sort=fold,
            init=init,
            tag=self.tag,
            loss='binary_crossentropy',
            verbose=True,
            ref=ref,
            model=get_model(self.b0, self.i0),
            output_dir=str(output_dir),
            batch_size=self.batch_size,
            dry_run=self.dry_run,
            et_bin=(et_bin_left, et_bin_right),
            eta_bin=(eta_bin_left, eta_bin_right)
        )
        logger.info(
            f'Training completed for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}')

    def run(self):
        logger = get_logger()
        dataset = ParquetDataset(dataset_dir=str(self.dataset_dir))
        n_folds = get_n_folds(
            kfold_table_glob=dataset.get_table_glob(self.kfold_table),
            fold_col=self.fold_col)
        logger.info(f'The dataset has {n_folds} folds.')
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        et_bins_iterator = zip(
            self.et_bins[:-1],
            self.et_bins[1:],
        )
        eta_bins_iterator = zip(
            self.eta_bins[:-1],
            self.eta_bins[1:],
        )
        if self.slurm_config:
            executor = self.slurm_config.get_executor()
        bins_iterator = product(et_bins_iterator, eta_bins_iterator)
        for i, ((et_bin_left, et_bin_right), (eta_bin_left, eta_bin_right)) in enumerate(bins_iterator):
            if i > 0 and self.dry_run:
                logger.info('Dry run enabled, stopping after first bin.')
                break
            for fold, init in product(folds_range, inits_range):
                if self.slurm_config is None:
                    self.run_training(
                        et_bin_left=et_bin_left,
                        et_bin_right=et_bin_right,
                        eta_bin_left=eta_bin_left,
                        eta_bin_right=eta_bin_right,
                        fold=fold,
                        init=init
                    )
                else:
                    logger.info('Submitting jobs to Slurm cluster...')
                    executor.submit(
                        self.run_training,
                        et_bin_left,
                        et_bin_right,
                        eta_bin_left,
                        eta_bin_right,
                        fold,
                        init
                    )
        logger.info('All training jobs submitted.')


app = typer.Typer(
    help='Ringer Zero VQAT commands'
)


@app.command()
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            '--config',
            help='Path to the YAML configuration file for the training job'
        )
    ]
):
    job = VQATTrainingJob.from_yaml(config)
    job.run()
