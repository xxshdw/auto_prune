import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn


@tf.RegisterGradient("Sofplus_neuron")
def sign_grad(op, grad):
    x = op.inputs[0]
#    w = tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape), tf.abs(tf.reciprocal(x)))
    w = tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape),
                 tf.multiply(tf.abs(tf.reciprocal(x)), tf.sqrt(tf.abs(tf.reciprocal(x)))))
    return tf.multiply(tf.multiply(tf.math.softplus(x), grad), w)

@tf.RegisterGradient("LeakyReLu_neuron")
def sign_grad(op, grad):
    x = op.inputs[0]
#    w = tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape), tf.abs(tf.reciprocal(x)))
    w = tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape),
                 tf.multiply(tf.abs(tf.reciprocal(x)), tf.sqrt(tf.abs(tf.reciprocal(x)))))
    return tf.multiply(w, tf.where(x <= 0, grad, 0.1 * grad))


def binarize_neuron(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with g.gradient_override_map({"Identity": "Sofplus_neuron"}):
        i = tf.identity(x)
        ge = tf.greater_equal(x, 0)
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
        return step_func


@tf.RegisterGradient("Sofplus")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.multiply(tf.math.softplus(In), grad)


@tf.RegisterGradient("LeakyReLu")
def sign_grad(op, grad):
    x = op.inputs[0]
    return tf.where(x <= 0, grad, 0.1 * grad)


def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with g.gradient_override_map({"Identity": "Sofplus"}):
        i = tf.identity(x)
        ge = tf.greater_equal(x, 0)
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
        return step_func


class MaskConv(Layer):
    def __init__(self, filters, kernel_size, padding, kernel_regularizer, strides=1, dilation_rate=1, **kwargs):
        super(MaskConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.input_spec = InputSpec(ndim=4)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')

    def build(self, input_shape):
        input_dim = input_shape[-1].value
        self.input_spec = [InputSpec(shape=input_shape)]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        # print(kernel_shape)
        self.weight = self.add_variable(self.name + "kernel", shape=kernel_shape,
                                        initializer='glorot_uniform',
                                        trainable=True,
                                        regularizer=self.kernel_regularizer)
        self.bias = self.add_variable(self.name + "bias", shape=[self.filters],
                                      initializer='zeros',
                                      trainable=True)
        self.mask = self.add_variable(self.name + "_mask", shape=kernel_shape,
                                      initializer=tf.keras.initializers.truncated_normal(mean=0.5, stddev=0.1),
                                      trainable=True)
        self.built = True

    def call(self, input):
        M_bin = binarize(self.mask)
        M_bin = tf.identity(M_bin, name=self.name + "_binary")
        kernel_bin = tf.multiply(self.weight, M_bin)
        outputs = tf.keras.backend.conv2d(input, kernel_bin, strides=(1, 1), padding=self.padding)
        return tf.keras.backend.bias_add(outputs, self.bias)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])


class MaskDense(Layer):
    def __init__(self, unit, kernel_regularizer, **kwargs):
        super(MaskDense, self).__init__()
        self.units = unit
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        input_dim = input_shape[-1].value
        self.weight = self.add_variable(self.name + "kernel", shape=[input_shape[-1].value, self.units],
                                        initializer='glorot_uniform',
                                        trainable=True,
                                        regularizer=self.kernel_regularizer)
        self.bias = self.add_variable(self.name + "bias", shape=[self.units],
                                      initializer='zeros',
                                      trainable=True)
        self.mask = self.add_variable(self.name + "_mask", shape=[input_shape[-1].value, self.units],
                                      initializer=tf.keras.initializers.truncated_normal(mean=0.5, stddev=0.1),
                                      trainable=True)
        self.built = True

    def call(self, input):
        M_bin = binarize(self.mask)
        M_bin = tf.identity(M_bin, name=self.name + "_binary")
        W_bin = tf.multiply(M_bin, self.weight)
        outputs = gen_math_ops.mat_mul(input, W_bin)
        # outputs = tf.keras.backend.conv2d(input, kernel_bin, strides=(1, 1), padding=self.padding)
        return nn.bias_add(outputs, self.bias)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, cat):
        super(MaskLayer, self).__init__()
        self.cat = cat

    def build(self, input_shape):
        self.kernel = self.add_variable(self.cat + "_mask", shape=[int(input_shape[-1])],
                                        initializer=tf.keras.initializers.truncated_normal(mean=0.5, stddev=0.2),
                                        trainable=True)

    def call(self, input):
        In = tf.identity(input, name=self.cat + "_bin_input")
        M_bin = binarize_neuron(self.kernel)
        M_bin = tf.identity(M_bin, name=self.cat + "_binary")
        return tf.multiply(In, M_bin)




