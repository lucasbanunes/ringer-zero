#!/usr/bin/env python
from itertools import product
from pathlib import Path
from typing import Annotated
import pandas as pd
import numpy as np
from keras import Sequential, Model
import hgq.layers as qlayers
from hgq.config import QuantizerConfigScope, QuantizerConfig
from hgq.constraints import Constant
import typer

from ringer_zero.tunning import training
from ringer_zero import logger


# def get_model(b0, i0, iq_b0, iq_i0) -> Model:
#     with (
#         QuantizerConfigScope(
#             place='all',
#             q_type='kbi',
#             k0=True,
#             b0=b0,
#             i0=i0,
#             bc=Constant(b0),
#             ic=Constant(i0)
#         )
#     ):
#         model = Sequential([
#             qlayers.QDense(
#                 5, 
#                 activation='relu', 
#                 input_shape=(50,),
#                 iq_conf=QuantizerConfig(
#                     q_type='kbi',
#                     k0=True,
#                     b0=iq_b0,
#                     i0=iq_i0,
#                     bc=Constant(iq_b0),
#                     ic=Constant(iq_i0)
#                 )
#             ),
#             qlayers.QDense(1, activation='sigmoid'),
#         ])

#     return model

def get_model(b0, i0, iq_b0=None, iq_i0=None) -> Model:
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
            qlayers.QDense(5, activation='relu', input_shape=(50,), iq_conf=None),
            qlayers.QDense(1, activation='sigmoid'),
        ])

    return model

def quantizer(rings):
    
    rings_q15 = np.round(rings * 32768)
    rings_q15 = np.clip(rings_q15, -32768, 32767).astype(np.int16)
    rings_q15 = rings_q15.astype(np.float32) / 32768.0

    return rings_q15

def data_loader(path, cv, sort):

    # for new training, we selected 1/2 of rings in each layer
    # pre-sample - 8 rings
    # EM1 - 64 rings
    # EM2 - 8 rings
    # EM3 - 8 rings
    # Had1 - 4 rings
    # Had2 - 4 rings
    # Had3 - 4 rings
    prefix = 'trig_L2_cl_ring_%i'

    # rings presmaple
    presample = [prefix % iring for iring in range(8//2)]

    # EM1 list
    sum_rings = 8
    em1 = [prefix % iring for iring in range(sum_rings, sum_rings+(64//2))]

    # EM2 list
    sum_rings = 8+64
    em2 = [prefix % iring for iring in range(sum_rings, sum_rings+(8//2))]

    # EM3 list
    sum_rings = 8+64+8
    em3 = [prefix % iring for iring in range(sum_rings, sum_rings+(8//2))]

    # HAD1 list
    sum_rings = 8+64+8+8
    had1 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

    # HAD2 list
    sum_rings = 8+64+8+8+4
    had2 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

    # HAD3 list
    sum_rings = 8+64+8+8+4+4
    had3 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

    pidname = 'el_lhmedium'
    ring_cols = presample + em1 + em2 + em3 + had1 + had2 + had3

    df = pd.read_hdf(path)
    df = df.loc[((df[pidname]) & (df.target == 1.0)) |
                ((df.target == 0) & (~df['el_lhvloose']))]

    rings = df[ring_cols].values.astype(np.float32)

    def norm1(data):
        norms = np.abs(data.sum(axis=1))
        norms[norms == 0] = 1
        return data/norms[:, None]

    target = df['target'].values.astype(np.int16)
    rings = norm1(rings)
    q_rings = quantizer(rings)
    splits = [(train_index, val_index)
              for train_index, val_index in cv.split(q_rings, target)]
    
    return q_rings[splits[sort][0]], q_rings[splits[sort][1]], target[splits[sort][0]], target[splits[sort][1]]


app = typer.Typer(
    help='Ringer Zero VQAT commands'
)


@app.command()
def run_training(
    datapath: Annotated[
        Path,
        typer.Option(
            '--datapath'
        )
    ],
    et: Annotated[
        int,
        typer.Option(
            '--et'
        )
    ],
    eta: Annotated[
        int,
        typer.Option(
            '--eta'
        )
    ],
    ref: Annotated[
        Path,
        typer.Option(
            '--ref'
        )
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            '--output-dir'
        )
    ],
    tag: Annotated[
        str,
        typer.Option(
            '--tag'
        )
    ],
    b0: Annotated[
        int,
        typer.Option(
            '--b0'
        )
    ],
    i0: Annotated[
        int,
        typer.Option(
            '--i0'
        )
    ],
    iq_b0: Annotated[
        int,
        typer.Option(
            '--iq-b0'
        )
    ],
    iq_i0: Annotated[
        int,
        typer.Option(
            '--iq-i0'
        )
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            '--batch-size'
        )
    ] = 1024,
    sorts: Annotated[
        int,
        typer.Option(
            '--sorts'
        )
    ] = 10,
    inits: Annotated[
        int,
        typer.Option(
            '--inits'
        )
    ] = 5,
    seed: Annotated[
        int,
        typer.Option(
            '--seed'
        )
    ] = 512,
    dry_run: Annotated[
        bool,
        typer.Option(
            '--dry-run'
        )
    ] = False
):
    if dry_run:
        sorts_range = [0]
        init_range = [0]
    else:
        sorts_range = range(sorts)
        init_range = range(inits)

    for i, (sort, init) in enumerate(product(sorts_range, init_range)):
        logger.info(
            f'{i} - Running sort {sort} and init {init}'
        )
        training(
            sort=sort,
            init=init,
            seed=seed,
            tag=tag,
            loss='binary_crossentropy',
            verbose=True,
            ref=str(ref),
            model=get_model(b0, i0, iq_b0, iq_i0),
            datapath=str(datapath),
            et=et,
            eta=eta,
            output_dir=str(output_dir),
            data_loader=data_loader,
            batch_size=batch_size,
            dry_run=dry_run
        )
