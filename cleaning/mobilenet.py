import warnings
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.keras.layers import Dense
from mobilenet_model import MobileNetV2
from numpy import prod
import numpy as np



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

 
#from tensorflow.keras.layers import Conv2D


ch = 3
n_output = 10
z = 4
# Amplifier for mask hyper parameters
k = 0.1
# Hyper Param
# Regular learning rate
lr = 1e-5  #  1e-5
# Sparsify learning rate
lr_sparse = 1.5e-2 # 1e-2
# Sparsify hyper parameters
lamda_sparse_fc = 0.06 # 0.02
lamda_sparse_cnn = 1 # 0.01
lamda_list = [0.1*lamda_sparse_cnn, lamda_sparse_cnn]

# Negative penalty should smaller than positive penalty
n_ratio = 0.001
batch_size = 100

# Common L1 norm hyper parameters
lamda_l1 = 1e-6 # 1e-5, 1e-6, 1e-7
# Common L2 norm hyper parameters
lamda_l2 = 0.0005 #5e-4

tf.reset_default_graph()

res = MobileNetV2(include_top=True, weights='imagenet')

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_x')
y_true = tf.placeholder(tf.float32, [None,n_output], name='y_true')
# Keep probability for dropout 
prob = tf.placeholder_with_default(0.8, shape=())
# Scheduling placeholder 
regu_rate = tf.placeholder(tf.float32, shape=[])
# Read Resnet Model
y = res(x)

# Get Mask Variables
all_mask_variables = [v for v in tf.global_variables() if "mask" in v.name and "Adam" not in v.name]
cnn_mask_variables = [v for v in all_mask_variables if "kernel" not in v.name and "bias" not in v.name and "conv" in v.name]
fc_mask_variables = [v for v in all_mask_variables if "kernel" not in v.name and "bias" not in v.name and "conv" not in v.name]

all_binary_layers = [t for tensor in tf.get_default_graph().get_operations() for t in tensor.values() if "binary" in tensor.name ]
cnn_mask_layers = [v for v in all_binary_layers if "conv" in v.name][:53]
cnn_mask_shape = [v.get_shape().as_list() for v in cnn_mask_layers]
fc_mask_layers = [v for v in all_binary_layers if "dense" in v.name][0]
fc_mask_shape = 4096000#fc_mask_layers[0].get_shape().as_list()
#fc_mask_shape = fc_mask_layers.get_shape().as_list()
#len(fc_mask_layers)
bin_all_cnn_list =  list(map(tf.reduce_sum, cnn_mask_layers))
bin_all_cnn = tf.cast(sum(bin_all_cnn_list), tf.float32)
n_total_cnn = sum(map(prod, cnn_mask_shape))
bin_percent_cnn = bin_all_cnn/n_total_cnn

bin_all_fc_list =  list(map(tf.reduce_sum, [fc_mask_layers]))
bin_all_fc = tf.cast(sum(bin_all_fc_list), tf.float32)
n_total_fc = 2048*1000+1000 #sum(map(sum, [fc_mask_shape]))
bin_percent_fc = bin_all_fc/n_total_fc
bin_percent = (bin_all_fc+bin_all_cnn)/(n_total_fc+n_total_cnn)

# Collect all weights 
all_weights = [var for var in tf.global_variables() if "kernel" in var.name and "Adam" not in var.name]
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
optimizer_acc = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8).minimize(cross_entropy_acc, var_list=[all_weights, all_bias]) 

optimizer_sp = tf.train.AdamOptimizer(learning_rate=regu_rate, epsilon=1e-8).minimize(cross_entropy_acc, var_list=[cnn_mask_variables, fc_mask_variables])

datagen_train = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
datagen = ImageDataGenerator()  # randomly flip images

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
    res.load_weights("resnet_kernel.h5")
    #writer = tf.summary.FileWriter("./output", sess.graph)
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
           
            print("step = {0:5d}, l_acc = {1:2.5f}, l_sp = {2:2.5f}, err_train = {6:2.2f}, err_test = {3:2.2f}, w_0/w = {4:2.2f}, X = {5:2.0f}"
                  .format(step, loss_acc, loss_sp, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100, 1/sparse_sparse[-1],(1-accuracy_sparse_before[-1])*100 ))
            cnn_per1, cnn_per2, fc_per = sess.run([bin_percent_cnn,bin_percent_cnn, bin_percent_fc] ,feed_dict={x: batch_x, y_true: batch_y})
            print("cnn1={0:2.2f}%, cnn2={1:2.2f}%, fc={2:2.2f}%".format(cnn_per1*100,cnn_per2*100, fc_per*100))
print(time.time() - init_time)
