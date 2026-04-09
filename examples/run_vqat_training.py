import logging
from pathlib import Path
from ringer_zero.models.vqat import run_training


def run_vqat_training(
    output_path: Path,
    datapath: str,
    ref: str,
    et: int,
    eta: int,
    b0: int,
    i0: int,
    iq_b0: int | None = None,
    iq_i0: int | None = None,
):
    output_dir = output_path / f"vqat-b{b0}-i{i0}_noluts/et{et}-eta{eta}"
    run_training(
        datapath=datapath,
        et=et,
        eta=eta,
        ref=ref,
        output_dir=str(output_dir),
        tag=f"vqat-b{b0}-i{i0}_noluts",
        b0=b0,
        i0=i0,
        iq_b0=None,
        iq_i0=None,
        batch_size=1024,
        sorts=10,
        inits=5,
        seed=512,
    )
    logging.info(f"Results saved in {output_dir}")


if __name__ == "__main__":
    run_vqat_training(
        output_path=Path("results/vqat"),
        datapath="/path/to/data/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5",
        ref="references/mc21_13p6TeV.Run3_v1.40bins.ref.json",
        et=5,
        eta=0,
        b0=22,
        i0=7,
        iq_b0=None,
        iq_i0=None,
    )
