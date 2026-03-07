import logging
from pathlib import Path
from ringer_zero.models.vnoq0 import run_training

def run_vnoq0_training(output_path: Path, 
                        datapath: str, ref: str, 
                        et: int, eta: int):
    output_dir = output_path / f'et{et}-eta{eta}'
    run_training(
        datapath=datapath,
        et=et,
        eta=eta,
        ref=ref,
        output_dir=str(output_dir),
        tag=f'vnoq0',
        batch_size=1024,
        sorts=10,
        inits=5,
        seed=512
    )
    logging.info(f'Results saved in {output_dir}')


if __name__ == '__main__':

    run_vnoq0_training(
        output_path=Path('results/vnoq0'),
        datapath='/home/gabri/Workspace/cern/data/2sigma_h5/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5',
        ref='references/mc21_13p6TeV.Run3_v1.40bins.ref.json',
        et=5,
        eta=0,
    )

