from __future__ import print_function
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_layer import InputSpec
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from numpy import prod
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
#from tensorflow.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from functools import reduce
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
import time


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
        self.strides=conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate=conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        
    def build(self, input_shape):
        input_dim = input_shape[-1].value
        self.input_spec = [InputSpec(shape=input_shape)]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        #print(kernel_shape)
        self.weight = self.add_variable(self.name+"kernel", shape=kernel_shape, 
                                        initializer='glorot_uniform',
                                        trainable = True,
                                        regularizer = self.kernel_regularizer)
        self.bias = self.add_variable(self.name+"bias", shape=[self.filters], 
                                        initializer='zeros',
                                        trainable = True)
        self.mask = self.add_variable(self.name+"_mask", shape=kernel_shape,
                                       initializer=tf.keras.initializers.truncated_normal(mean=0.5,stddev=0.1),
                                       trainable = True)
        self.built = True
        
    def call(self, input):
        M_bin = binarize(self.mask) 
        M_bin = tf.identity(M_bin, name=self.name+"_binary")  
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
        return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])

    
class MaskDense(Layer):
    def __init__(self, unit, kernel_regularizer, **kwargs):
        super(MaskDense, self).__init__()
        self.units = unit
        self.kernel_regularizer = kernel_regularizer
    
    def build(self, input_shape):
        input_dim = input_shape[-1].value
        self.weight = self.add_variable(self.name+"kernel", shape=[input_shape[-1].value, self.units], 
                                        initializer='glorot_uniform',
                                        trainable = True,
                                        regularizer = self.kernel_regularizer)
        self.bias = self.add_variable(self.name+"bias", shape=[self.units], 
                                        initializer='zeros',
                                        trainable = True)
        self.mask = self.add_variable(self.name+"_mask", shape=[input_shape[-1].value, self.units],
                                       initializer=tf.keras.initializers.truncated_normal(mean=0.5,stddev=0.1),
                                       trainable = True)
        self.built = True
    def call(self, input):
        M_bin = binarize(self.mask) 
        M_bin = tf.identity(M_bin, name=self.name+"_binary")  
        W_bin = tf.multiply(M_bin, self.weight)
        outputs = gen_math_ops.mat_mul(input, W_bin)
        #outputs = tf.keras.backend.conv2d(input, kernel_bin, strides=(1, 1), padding=self.padding)
        return nn.bias_add(outputs, self.bias)
    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
              raise ValueError(
                  'The innermost dimension of input_shape must be defined, but saw: %s'
                  % input_shape)
        return input_shape[:-1].concatenate(self.units)
    

    
def build_model(lamda_l1, lamda_l2):
    
    model = Sequential()
    weight_decay = 0.0005

    model.add(MaskConv(64, (3, 3), input_shape=[32,32,3], padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(MaskConv(64, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(128, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(128, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(MaskConv(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(MaskDense(512,kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(MaskDense(10,kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('softmax'))
    
    return model

def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def normalize_production(x):
    #this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.

    #these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

#try zero neg gradient
@tf.RegisterGradient("Sofplus")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.multiply(tf.math.softplus(In),grad)

@tf.RegisterGradient("LeakyReLu")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.where(In<=0, grad, 0.1*grad)

image_size = 32
ch = 3
n_output = 10
z = 4
# Amplifier for mask hyper parameters
k = 0.1
# Hyper Param
# Regular learning rate
lr = 1e-5  #  1.5e-5
# Sparsify learning rate
lr_sparse = 5e-3 # 1.5e-2
# Sparsify hyper parameters
lamda_sparse_fc = 0#0.06#0.1#0.000005 # 0.06
lamda_sparse_cnn = 0#0.8#1#0.00001 # 1
n_ratio = 0.001#0.001
lamda_list = [0.1*lamda_sparse_cnn, lamda_sparse_cnn]

# Negative penalty should smaller than positive penalty
batch_size = 200

# Common L1 norm hyper parameters
lamda_l1 = 1e-6#1e-7 # 1e-5, 1e-6, 1e-7
# Common L2 norm hyper parameters
lamda_l2 = 5e-4#5e-4 #5e-4

tf.reset_default_graph()

vgg16 = build_model(lamda_l1, lamda_l2)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input_x')
y_true = tf.placeholder(tf.float32, [None,n_output], name='y_true')
# Keep probability for dropout 
prob = tf.placeholder_with_default(0.8, shape=())
# Scheduling placeholder 
regu_rate = tf.placeholder(tf.float32, shape=[])
# Read VGG16 Model
y = vgg16(x)


# Get Mask Variables
all_mask_variables = [v for v in tf.global_variables() if "mask" in v.name and "Adam" not in v.name]
cnn_mask_variables = [v for v in all_mask_variables if "kernel" not in v.name and "bias" not in v.name and "conv" in v.name]
fc_mask_variables = [v for v in all_mask_variables if "kernel" not in v.name and "bias" not in v.name and "conv" not in v.name]

all_binary_layers = [t for tensor in tf.get_default_graph().get_operations() for t in tensor.values() if "binary" in tensor.name ]
cnn_mask_layers = [v for v in all_binary_layers if "conv" in v.name][:13]
cnn_mask_shape = [v.get_shape().as_list() for v in cnn_mask_layers]
fc_mask_layers = [v for v in all_binary_layers if "dense" in v.name][:2]
#fc_mask_shape = fc_mask_layers.get_shape().as_list()
#len(fc_mask_layers)
bin_all_cnn_list =  list(map(tf.reduce_sum, cnn_mask_layers))
bin_all_cnn = tf.cast(sum(bin_all_cnn_list), tf.float32)
n_total_cnn = sum(map(prod, cnn_mask_shape))
bin_percent_cnn = bin_all_cnn/n_total_cnn

bin_all_fc_list =  list(map(tf.reduce_sum, fc_mask_layers))
bin_all_fc = tf.cast(sum(bin_all_fc_list), tf.float32)
n_total_fc = 512*512+512*10 #sum(map(sum, [fc_mask_shape]))
bin_percent_fc = bin_all_fc/n_total_fc
bin_percent = (bin_all_fc+bin_all_cnn)/(n_total_fc+n_total_cnn)

# Collect all weights 
all_weights = [var for var in tf.global_variables() if "kernel" in var.name and "Adam" not in var.name]
all_bias = [var for var in tf.global_variables() if "bias" in var.name and "Adam" not in var.name]

pos_penalty = lamda_sparse_fc*bin_percent_fc + lamda_sparse_cnn*bin_percent_cnn
# Cross Entropy Loss, 2-Norm is handeled by the model
# L = L + ||w||1 + ||w||2 + ||bin_cnn|| + ||bin_fc|| + ||m||2
cross_entropy_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y)) 
regu_loss = pos_penalty

cross_entropy_2 = cross_entropy_acc# + regu_loss
# Define Optimizer tf.contrib.opt.NadamOptimizer
optimizer1 =  tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
gradients1, vriables1 = zip(*optimizer1.compute_gradients(cross_entropy_acc, var_list=[all_weights, all_bias]))
#abs_invers_var = [tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0.,shape=x.shape), tf.multiply(tf.abs(tf.reciprocal(x)),tf.sqrt(tf.abs(tf.reciprocal(x))))) for x in vriables1]
abs_invers_var = [tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0.,shape=x.shape), tf.abs(tf.reciprocal(x))) for x in vriables1]

optimizer2 = tf.train.AdamOptimizer(learning_rate=regu_rate, epsilon=1e-8)
gradients2, vriables2 = zip(*optimizer2.compute_gradients(cross_entropy_2, var_list=[cnn_mask_variables, fc_mask_variables]))
boost_gradient2 = [tf.multiply(x,y) for x,y in zip(gradients2,abs_invers_var[:15])]
optimizer_sp = optimizer2.apply_gradients(zip(boost_gradient2, vriables2))


optimizer1 =  tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
gradients1, vriables1 = zip(*optimizer1.compute_gradients(cross_entropy_acc, var_list=[all_weights, all_bias]))
optimizer_acc = optimizer1.apply_gradients(zip(gradients1, vriables1))

datagen_test = ImageDataGenerator()  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).


saver = tf.train.Saver()
init = tf.global_variables_initializer()
accuracy_sparse_before = [0]
accuracy_sparse = [0]
sparse_sparse = [0]
percent = 1
init_time = time.time()
check = 0
check_increase = []
counter = nper = 0
with tf.Session() as sess:
    sess.run(init)
    vgg16.load_weights("vgg_model_kernel_init.h5")
    #writer = tf.summary.FileWriter("./output", sess.graph)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
    y_train = keras.utils.to_categorical(y_train, n_output)
    y_test = keras.utils.to_categorical(y_test, n_output)
    train_gen = datagen_test.flow(x=x_train, y=y_train, batch_size=batch_size)
    test_gen = datagen_test.flow(x=x_test, y=y_test, batch_size=2000)
    
    for step in range(200000):  
        batch_x, batch_y = train_gen.next()

        _, loss_acc, par = sess.run([optimizer_acc,cross_entropy_acc,cnn_mask_variables] ,feed_dict={x: batch_x, y_true: batch_y})         
        _,  percent, vv, aa, bb, cc, loss_sp = sess.run([optimizer_sp, bin_percent, vriables1, gradients1, boost_gradient2, vriables2, cross_entropy_acc] ,feed_dict={x: batch_x, y_true: batch_y,regu_rate:lr_sparse})

        if check < percent:
            counter = counter + 1
            nper += percent - check
        
        check = percent

        sparse_sparse.append(percent)
        if step%500 == 0:
            print(counter)
            check_increase.append(counter) 
            counter = 0 
            nper = 0
            batch_test_x, batch_test_y = test_gen.next()
            matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
            evaluate = tf.reduce_mean(tf.cast(matches,tf.float32))
            accuracy_sparse_before.append(sess.run(evaluate,feed_dict={x:batch_x, y_true:batch_y, prob:1}))
            accuracy_sparse.append(sess.run(evaluate,feed_dict={x:batch_test_x, y_true:batch_test_y,prob:1}))
            else:
                print("step = {0:5d}, l_acc = {1:2.5f}, l_sp = {2:2.5f}, err_train = {6:2.2f}, err_test = {3:2.2f}, w_0/w = {4:2.2f}, X = {5:2.0f}"
                      .format(step, loss_acc, loss_sp, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100, 1/sparse_sparse[-1],(1-accuracy_sparse_before[-1])*100 ))
                cnn_per1, cnn_per2, fc_per = sess.run([bin_percent_cnn,bin_percent_cnn, bin_percent_fc] ,feed_dict={x: batch_x, y_true: batch_y})
                print("cnn1={0:2.2f}%, cnn2={1:2.2f}%, fc={2:2.2f}%".format(cnn_per1*100,cnn_per2*100, fc_per*100))
    #writer.close()               
print(time.time() - init_time)