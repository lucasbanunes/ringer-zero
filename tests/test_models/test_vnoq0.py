import logging
from pathlib import Path
from ringer_zero.models.vnoq0 import run_training


def test_run_training(tmp_path: Path):
    output_dir = tmp_path / 'output' / 'vnoq0_test'
    run_training(
        datapath='/media/lucasbanunes/KINGSTON/data/cern_data/isabela/qt_data_mc21_5m/2sigma_h5/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5',
        et=5,
        eta=0,
        ref='/media/lucasbanunes/KINGSTON/data/cern_data/isabela/Models_Scripts/references/mc21_13p6TeV.Run3_v1.40bins.ref.json',
        output_dir=str(output_dir),
        tag='vnoq0-test',
        batch_size=1024,
        sorts=1,
        inits=1,
        seed=512,
        dry_run=True
    )
    logging.info(f'Results saved in {output_dir}')
