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
#from tensorflow.keras.layers import Conv2D

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, padding='valid',rank=2, kernel_regularizer=None, strides=1, dilation_rate=1, **kwargs):
        super(Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.padding = conv_utils.normalize_padding(padding)
        self.kernel_regularizer = kernel_regularizer
        self.input_spec = InputSpec(ndim=4)
        self.strides=conv_utils.normalize_tuple(strides, rank, 'strides')
        self.dilation_rate=conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        
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
        outputs = tf.keras.backend.conv2d(input, kernel_bin, strides=self.strides, padding=self.padding)
        return tf.keras.backend.bias_add(outputs, self.bias)

class Dense(Layer):
    def __init__(self, unit, kernel_regularizer=None, **kwargs):
        super(Dense, self).__init__()
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

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = MaskLayer(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        #model.load_weights(weights_path)
        

    
    return model

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

res = ResNet50(include_top=True, weights='imagenet')

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
    
    for step in range(200000):  
        batch_x, batch_y = datagen_train.next()
       
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
