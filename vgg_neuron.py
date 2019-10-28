from tensorflow.keras.layers import Layer
import tensorflow as tf
from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
#from tensorflow.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from functools import reduce
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

    
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, cat):
        super(MaskLayer, self).__init__()
        self.cat = cat
        
    def build(self, input_shape):
        self.kernel = self.add_variable(self.cat+"_mask", shape=[int(input_shape[-1])],
                                       initializer=tf.keras.initializers.truncated_normal(mean=0.5,stddev=0.1),
                                       trainable = True)
    
    def call(self, input):
        M_bin = binarize(self.kernel) 
        M_bin = tf.identity(M_bin, name=self.cat+"_binary")  
        return tf.multiply(input, M_bin)

def build_model(lamda_l1, lamda_l2):
    
    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=[32,32,3],kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn11"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn12"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn21"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn22"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn31"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn32"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn33"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn41"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn42"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn43"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn51"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn52"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn53"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(MaskLayer("fc0"))
    model.add(Dense(512,kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("fc1"))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
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

@tf.RegisterGradient("Linear")
def sign_grad(op, grad):
    return grad

#try zero neg gradient
@tf.RegisterGradient("Sofplus")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.multiply(tf.math.softplus(In),grad)

@tf.RegisterGradient("ReLu")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.where(In>0, grad, tf.zeros(grad.shape))

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
lr = 1e-5  #  1e-5
# Sparsify learning rate
lr_sparse = 1e-2 # 1e-2
# Sparsify hyper parameters
lamda_sparse_fc = 0.01 # 0.02
lamda_sparse_cnn = 0.1 # 0.01
lamda_list = [0.1*lamda_sparse_cnn, lamda_sparse_cnn]

# Negative penalty should smaller than positive penalty
n_ratio = 0.001
batch_size = 200

# Common L1 norm hyper parameters
lamda_l1 = 1e-6 # 1e-5, 1e-6, 1e-7
# Common L2 norm hyper parameters
lamda_l2 = 0.0005 #5e-4
vgg16 = build_model(lamda_l1, lamda_l2)


#tf.reset_default_graph()
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
cnn_mask_variables = [v for v in all_mask_variables if "cnn" in v.name]
fc_mask_variables = [v for v in tf.global_variables() if "fc" in v.name]
# Get Binary Tensors
all_binary_layers = [t for tensor in tf.get_default_graph().get_operations() for t in tensor.values() if "binary" in tensor.name ]
cnn_mask_layers = [v for v in all_binary_layers if "cnn" in v.name]
cnn_mask_shape = [v.get_shape().as_list() for v in cnn_mask_layers]
fc_mask_layers = [v for v in all_binary_layers if "fc" in v.name]
fc_mask_shape = [v.get_shape().as_list() for v in fc_mask_layers]

bin_all_cnn_list =  list(map(tf.reduce_sum, cnn_mask_layers))
bin_all_cnn = tf.cast(sum(bin_all_cnn_list), tf.float32)
n_total_cnn = sum(map(sum, cnn_mask_shape))
bin_percent_cnn = bin_all_cnn/n_total_cnn

bin_all_fc_list =  list(map(tf.reduce_sum, fc_mask_layers))
bin_all_fc = tf.cast(sum(bin_all_fc_list), tf.float32)
n_total_fc = sum(map(sum, fc_mask_shape))
bin_percent_fc = bin_all_fc/n_total_fc
bin_percent = (bin_all_fc+bin_all_cnn)/(n_total_fc+n_total_cnn)

# Collect all weights 
#all_weights = [var for var in tf.global_variables() if "kernel" in var.name and "Adam" not in var.name]
all_weights = [var for var in tf.global_variables() if ("kernel" in var.name or "batch_normalization" in var.name) and "Adam" not in var.name]
all_bias = [var for var in tf.global_variables() if "bias" in var.name and "Adam" not in var.name]

pos_penalty = lamda_sparse_fc*bin_percent_fc + lamda_sparse_cnn*bin_percent_cnn
# Cross Entropy Loss, 2-Norm is handeled by the model
# L = L + ||w||1 + ||w||2 + ||bin_cnn|| + ||bin_fc|| + ||m||2
cross_entropy_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y)) 
cross_entropy_acc += pos_penalty

# Adding neg penality
nRegu = [binarize(tf.negative(x)) for x in fc_mask_variables]
nRegu_cnn = [binarize(tf.negative(x)) for x in cnn_mask_variables]
nbin_all_fc = sum(list(map(tf.reduce_sum, nRegu)))
nbin_all_cnn =  sum(list(map(tf.reduce_sum, nRegu_cnn)))
neg_penalty = lamda_sparse_fc*n_ratio*(nbin_all_fc/n_total_fc) + lamda_sparse_cnn*n_ratio*(nbin_all_cnn/n_total_cnn)
cross_entropy_acc += neg_penalty

# Define Optimizer tf.contrib.opt.NadamOptimizer
optimizer_acc = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-6).minimize(cross_entropy_acc, var_list=[all_weights, all_bias]) 

optimizer_sp = tf.train.AdamOptimizer(learning_rate=regu_rate, epsilon=1e-8).minimize(cross_entropy_acc, var_list=[cnn_mask_variables, fc_mask_variables])

datagen = ImageDataGenerator()  

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
    vgg16.load_weights("vgg_model.h5")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
    y_train = keras.utils.to_categorical(y_train, n_output)
    y_test = keras.utils.to_categorical(y_test, n_output)
    train_gen = datagen.flow(x=x_train, y=y_train, batch_size=batch_size)
    test_gen = datagen.flow(x=x_test, y=y_test, batch_size=2000)
    
    for step in range(200000):  
        batch_x, batch_y = train_gen.next()

        _, loss_acc, par = sess.run([optimizer_acc, cross_entropy_acc, cnn_mask_variables] ,feed_dict={x: batch_x, y_true: batch_y})         
        _, percent, loss_sp = sess.run([optimizer_sp, bin_percent, cross_entropy_acc] ,feed_dict={x: batch_x, y_true: batch_y,regu_rate:lr_sparse})
        
        if check < percent:
            counter = counter + 1
            nper += percent - check
        
        check = percent
       
        sparse_sparse.append(percent)
        if step%1000 == 0:
            print(counter)
            check_increase.append(counter) 
            counter = 0 
            nper = 0
            batch_test_x, batch_test_y = test_gen.next()
            matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
            evaluate = tf.reduce_mean(tf.cast(matches,tf.float32))
            accuracy_sparse_before.append(sess.run(evaluate,feed_dict={x:batch_x, y_true:batch_y, prob:1}))
            accuracy_sparse.append(sess.run(evaluate,feed_dict={x:batch_test_x, y_true:batch_test_y,prob:1}))
            
            if 0:
                print("step = {0:5d}, l_acc = {1:2.5f}, err = {2:2.2f}, w_0/w = {3:2.2f}"
                      .format(step, loss_acc, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100))
            else:
                print("step = {0:5d}, l_acc = {1:2.5f}, l_sp = {2:2.5f}, err_train = {6:2.2f}, err_test = {3:2.2f}, w_0/w = {4:2.2f}, X = {5:2.0f}"
                      .format(step, loss_acc, loss_sp, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100, 1/sparse_sparse[-1],(1-accuracy_sparse_before[-1])*100 ))
                cnn_per1, cnn_per2, fc_per = sess.run([bin_percent_cnn,bin_percent_cnn, bin_percent_fc] ,feed_dict={x: batch_x, y_true: batch_y})
                print("cnn1={0:2.2f}%, cnn2={1:2.2f}%, fc={2:2.2f}%".format(cnn_per1*100,cnn_per2*100, fc_per*100))
    writer.close()               
print(time.time() - init_time)
