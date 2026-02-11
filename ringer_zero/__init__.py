# Based on: https://github.com/ringer-softwares/neuralnet
import colorlog
import logging
import pandas
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


#
# Set logger
#
logger = colorlog.getLogger()
logger.setLevel(logging.DEBUG)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s [%(asctime)s] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
logger.addHandler(handler)

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
    logger.error("Not possible to set gpu allow growth")
    raise


# bugfix: https://stackoverflow.com/questions/63158424/why-does-keras-model-fit-with-sample_weight-have-long-initialization-time
def class_weight(target):
    classes = np.unique(target)  # .tolist()
    # [-1,1] or [0,1]
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=target)
    # class_weights = {cl: weights[idx] for idx, cl in enumerate(classes)}
    sample_weight = np.ones_like(target, dtype=np.float32)
    sample_weight[target == 1] = weights[1]
    sample_weight[target != 1] = weights[0]
    return pandas.Series(sample_weight).to_frame('weight')


def check_batch_size(target, batch_size):
    _, n_evt_per_class = np.unique(target, return_counts=True)
    batch_size = (batch_size if np.min(n_evt_per_class) >
                  batch_size else np.min(n_evt_per_class))
    return batch_size
