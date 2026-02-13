
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from typing import Callable
from keras import Model

from .callbacks import sp
from .decorators import Summary, Reference
from . import check_batch_size, class_weight, logger


def training(
        sort: int,
        init: int,
        seed: int,
        tag: str,
        loss: str,
        verbose: bool,
        ref: str,
        model: Model,
        datapath: str,
        et: int,
        eta: int,
        output_dir: str,
        data_loader: Callable[[str, StratifiedKFold, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        batch_size: int,
        optimizer: str = 'adam',
        metrics: list[str] = ['accuracy'],
        callbacks: list[str] = [],
        patience: int = 25,
        detailed=False,
        dry_run: bool = False,
        **kw):

    output_dir = output_dir + \
        '/tuned.%s.sort_%d.init_%d.model' % (tag, sort, init)
    if os.path.exists(output_dir):
        print('Output already exists')
        return
    os.makedirs(output_dir, exist_ok=True)

    #
    # Create cross-validation sorts
    #
    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    #
    # Load data
    #
    data, data_val, target, target_val = data_loader(datapath, cv, sort)

    #
    # Compile the model
    #
    model.compile(optimizer, loss=loss, metrics=metrics)
    model.summary()

    with open(ref, 'r') as f:
        ref = json.load(f)
    decorators = [Summary(detailed=detailed),
                  Reference(ref[et][eta])]

    if len(callbacks) == 0:
        callbacks.append(
            sp(validation_data=(data_val, target_val),
                patience=patience,
                verbose=verbose,
                save_the_best=True
               )
        )
    #
    # Train model
    #
    start = datetime.now()
    # y_hat = f(x)
    history = model.fit(data, target,
                        epochs=5000 if dry_run else 5000,
                        batch_size=check_batch_size(target, batch_size),
                        verbose=verbose,
                        validation_data=(data_val, target_val),
                        sample_weight=class_weight(target),
                        callbacks=[callbacks] if not isinstance(
                            callbacks, list) else callbacks,
                        shuffle=True
                        ).history

    end = datetime.now()

    history['patience'] = patience

    logger.info(f"Training step: {end-start}")

    #
    # Loop over decorators
    #
    for decorator in decorators:
        decorator(history, {'model': model, 'data': (
            data, target),  'data_val': (data_val, target_val)})

    d = {'history': history,
         'model': json.loads(model.to_json()),
         'weights': model.get_weights(),
         'metadata': {
             'et_bin': et,
             'eta_bin': eta,
             'sort': sort,
             'init': init,
             'tag': tag,
         },
         'time': (end-start)}

    #
    # Save the file
    #

    with open(output_dir + '/results.pic', 'wb') as f:
        pickle.dump(d, f)

    # model.export(
    #     filepath=output_dir + '/model.onnx',
    #     format='onnx'
    # )
    model.export(
        filepath=output_dir + '/model.tf',
        format='tf_saved_model'
    )


def reprocessing(args, data_loader, detailed=False):

    tf.config.run_functions_eagerly(False)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)
    tf.config.run_functions_eagerly(False)

    tuned = pickle.load(open(args.inputFile, 'rb'))
    metadata = tuned['metadata']
    history = tuned['history']
    sort = metadata['sort']
    init = metadata['init']
    model_idx = metadata['model']
    et = metadata['et_bin']
    eta = metadata['eta_bin']
    time = tuned['time']
    seed = 512
    tag = metadata['tag']

    # reload ref
    ref = json.load(open(args.ref, 'r'))

    # reload model
    model = keras.models.model_from_json(
        json.dumps(tuned['model'], separators=(',', ':')))
    model.set_weights(tuned['weights'])
    model.summary()

    # reload cross validation
    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # reload data
    data, data_val, target, target_val = data_loader(args.dataFile, cv, sort)

    decorators = [Summary(detailed=detailed), Reference(ref[et][eta])]

    for decorator in decorators:
        decorator(history, {'model': model, 'data': (
            data, target),  'data_val': (data_val, target_val)})

    # save all
    d = {
        'history': history,
        'model': json.loads(model.to_json()),
        'weights': model.get_weights(),
        'time': time,
        'metadata': {
            'et_bin': et,
            'eta_bin': eta,
            'sort': sort,
            'init': init,
            'model': model_idx,
            'tag': tag,
        }
    }

    output = args.output if args.output is not None else args.volume + \
        '/tuned.%s.sort_%d.init_%d.model_%d.pic' % (tag, sort, init, model_idx)
    pickle.dump(d, open(output, 'wb'))
