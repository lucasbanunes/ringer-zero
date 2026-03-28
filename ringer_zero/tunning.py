
import gc
import os
import json
import pickle
import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from keras import Model
#
# Set GPU memory control
#
try:
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    tf.config.run_functions_eagerly(False)
except Exception:
    # traceback.print_exc()
    import colorlog
    logger = colorlog.getLogger()
    logger.error("Not possible to set gpu allow growth")
    raise

from .tensorflow.callbacks import SP
from .decorators import Summary, Reference
from . import check_batch_size, class_weight, logger


type RefType = dict[str, dict[str, dict[str, float]]]


def training(
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sort: int,
        init: int,
        tag: str,
        loss: str,
        verbose: bool,
        ref: RefType,
        model: Model,
        et_bin: tuple[float, float],
        eta_bin: tuple[float, float],
        output_dir: str,
        batch_size: int,
        optimizer: str = 'adam',
        metrics: list[str] = ['accuracy'],
        callbacks=None,
        patience: int = 25,
        detailed=False,
        dry_run: bool = False):

    if callbacks is None:
        callbacks = []

    output_dir = output_dir + \
        '/tuned.%s.sort_%d.init_%d.model' % (tag, sort, init)
    if os.path.exists(output_dir):
        print('Output already exists')
        return
    os.makedirs(output_dir, exist_ok=True)

    model.compile(optimizer, loss=loss, metrics=metrics)
    model.summary()

    decorators = [Summary(detailed=detailed),
                  Reference(ref)]

    if len(callbacks) == 0:
        callbacks.append(
            SP(validation_data=(X_val, y_val),
                patience=patience,
                verbose=verbose,
                save_the_best=True
               )
        )

    start = datetime.now()
    history = model.fit(X, y,
                        epochs=1 if dry_run else 5000,
                        batch_size=check_batch_size(y, batch_size),
                        verbose=verbose,
                        validation_data=(X_val, y_val),
                        sample_weight=class_weight(y),
                        callbacks=[callbacks] if not isinstance(
                            callbacks, list) else callbacks,
                        shuffle=True
                        ).history

    end = datetime.now()

    history['patience'] = patience

    logger.info(f"Training step: {end-start}")

    for decorator in decorators:
        decorator(
            history,
            {'model': model,
             'data': (X, y),
             'data_val': (X_val, y_val)})

    d = {'history': history,
         'model': json.loads(model.to_json()),
         'weights': model.get_weights(),
         'metadata': {
             'et_bin': [str(et) for et in et_bin],
             'eta_bin': [str(eta) for eta in eta_bin],
             'sort': sort,
             'init': init,
             'tag': tag,
         },
         'time': (end-start)}

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
    keras.backend.clear_session()
    del model
    gc.collect()
    return history


def reprocessing(args, data_loader, detailed=False):

    raise NotImplementedError(
        "This function is not implemented yet. It is a placeholder for the future implementation of the reprocessing step of the tuning process.")

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
