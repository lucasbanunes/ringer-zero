import logging
from pathlib import Path
from ringer_zero.models.vqat0 import run_training


def test_run_training(tmp_path: Path, test_data_dir: Path):
    output_dir = tmp_path / 'output' / 'vqat0_test'
    run_training(
        datapath=str(test_data_dir / 'test_data.hdf5'),
        et=5,
        eta=0,
        ref=str(test_data_dir / 'test_ref.json'),
        output_dir=str(output_dir),
        tag='vqat0-test',
        batch_size=1024,
        sorts=1,
        inits=1,
        seed=512,
        dry_run=True
    )
    logging.info(f'Results saved in {output_dir}')
