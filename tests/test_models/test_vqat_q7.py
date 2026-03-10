import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Desabilita logicamente todas as GPUs para o TensorFlow
tf.config.set_visible_devices([], 'GPU')

# Garante que o XLA não tente compilar para GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import logging
from pathlib import Path
from ringer_zero.models.vqat_q7 import run_training


def test_run_training(tmp_path: Path, test_data_dir: Path):
    output_dir = tmp_path / 'output' / 'vqat_q7_test'
    run_training(
        datapath=str(test_data_dir / 'test_data.hdf5'),
        et=5,
        eta=0,
        ref=str(test_data_dir / 'test_ref.json'),
        output_dir=str(output_dir),
        tag='vqat_q7-test',
        batch_size=1024,
        sorts=1,
        inits=1,
        seed=512,
        dry_run=True
    )
    logging.info(f'Results saved in {output_dir}')
    print("Treinamento concluído")
