from abc import ABC
from keras import activations, ops, initializers, Variable
from keras.layers import Layer, Dense, Activation

from . import straight_through_clip


class UniformQuantization(Layer, ABC):

    def __init__(self, bits: int, integer_bits: int, **kwargs):
        super(UniformQuantization, self).__init__(**kwargs)
        self.bits = self.add_weight(
            name="bits",
            shape=(),
            initializer=initializers.Constant(bits),
            trainable=False,
        )
        self.integer_bits = self.add_weight(
            name="integer_bits",
            shape=(),
            initializer=initializers.Constant(integer_bits),
            trainable=False,
        )
        self.floating_bits = self.bits - self.integer_bits - 1
        self.floating_power = 2 ** self.floating_bits
        self.inverse_floating_power = 2 ** (-self.floating_bits)
        self.upper_limit = (2 ** self.integer_bits) - \
            (2 ** (-self.floating_bits))
        self.lower_limit = -(2 ** self.integer_bits)

    def quantize_weights(self, weights: Variable) -> Variable:
        int_weights = ops.cast(weights*self.floating_power, dtype='int32')
        floating_weights = ops.cast(
            int_weights, dtype='float32') * self.inverse_floating_power
        quantized_weights = straight_through_clip(
            # Clip weights to the representable range
            floating_weights, self.lower_limit, self.upper_limit)
        return quantized_weights


class UniformQuantizationDense(UniformQuantization):
    def __init__(self,
                 bits: int,
                 integer_bits: int,
                 units: int,
                 activation: str | None = None,
                 **kwargs):
        super(UniformQuantizationDense, self).__init__(
            bits=bits,
            integer_bits=integer_bits,
            **kwargs
        )
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        # Create trainable weights for the layer
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="uniform",
            trainable=True,
        )
        if self.activation is not None:
            self.activation_fn = activations.get(self.activation)
        else:
            self.activation_fn = None

    def call(self, inputs):
        # Apply quantization to the kernel weights
        quantized_kernel = self.quantize_weights(self.kernel)
        output = ops.matmul(inputs, quantized_kernel)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output


type SuportedUniformQuantizationTypes = Dense | Activation

type UniformQuantizationLayerTypes = UniformQuantizationDense | Activation


def uniform_quantization_layer(layer: SuportedUniformQuantizationTypes, bits: int, integer_bits: int) -> UniformQuantizationLayerTypes:
    if isinstance(layer, Dense):
        layer = UniformQuantizationDense(
            bits=bits,
            integer_bits=integer_bits,
            units=layer.units,
            activation=layer.activation,
            name=f"quantized_{layer.name}"
        )
    elif isinstance(layer, Activation):
        # For activation layers, we don't quantize them directly
        return layer
    else:
        raise NotImplementedError(
            "Unsupported layer type for uniform quantization")

    layer.set_weights(layer.get_weights())
    return layer
