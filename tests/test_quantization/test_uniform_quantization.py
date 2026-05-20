import numpy as np
from keras import Input, Sequential
from keras.layers import Dense
import tensorflow as tf

from neuralnet.quantization.uniform import UniformQuantizationDense


def test_uniform_quantization_layer_predicts_with_sequential_model():
    tf.config.run_functions_eagerly(True)
    model = Sequential(
        [
            Input(shape=(4,)),
            UniformQuantizationDense(
                bits=23,
                integer_bits=7,
                units=3,
                activation="relu",
            ),
            Dense(1, activation="sigmoid"),
        ]
    )

    random_data = np.random.default_rng(0).random((100, 4), dtype=np.float32)*10 - 5

    predictions = model.predict(random_data, verbose=0)

    assert predictions.shape == (100, 1), f"Expected predictions shape to be (100, 1), but got {predictions.shape}"
    assert np.isfinite(predictions).all(), "Expected all predictions to be finite numbers, but found non-finite values"
