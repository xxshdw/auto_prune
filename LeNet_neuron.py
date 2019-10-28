import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from functools import reduce
import time

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

@tf.RegisterGradient("Sofplus")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.multiply(tf.math.softplus(In),grad)

@tf.RegisterGradient("ReLu")
def sign_grad(op, grad):
    In = op.inputs[0]
    return tf.where(In>0, grad, tf.zeros(grad.shape))

image_size = 28
n_input = image_size * image_size
ch = 1
kernel_size = 5
n_output = 10
layers_cnn = {}    # Dictionary for cnn parameters
layers_fc = {}     # Dictionary for fc parameters

# Network size
n_layer = [n_input, 300, 100, n_output]     # Lenet 300-100
n_layer_cnn = [ch, 20, 50]               # CNN 
n_layer_fc = [4*4*50, 500, n_output]    # FC after CNN.

def binarize_cnn(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with ops.name_scope("Linear") as name:
        with g.gradient_override_map({"Identity": "Sofplus"}):
            i = tf.identity(x)
            ge = tf.greater_equal(x, 0)
            # tf.stop_gradient is needed to exclude tf.to_float from derivative
            step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
            return step_func
#Identity
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Identity": "Sofplus"}):
            i = tf.identity(x)
            ge = tf.greater_equal(x, 0)
            # tf.stop_gradient is needed to exclude tf.to_float from derivative
            step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
            return step_func
        
        
def Multi_layer_fc(x, layer_config, param, sparse=False, dropout=False, name="FC"):
    
    y = {}
    
    input_layer = {'M':tf.Variable(tf.truncated_normal(shape=[layer_config[0]],mean=0.5,stddev=0.01, name='Ad_mat'))}
    param['M'].append(input_layer['M'])
    if sparse == True:
        param['binary_M'].append(binarize(input_layer['M']))
        x = tf.multiply(x, param['binary_M'][-1])
    
    with tf.name_scope(name):
        for i in range(1, len(layer_config)):
            new_layer = {'W': tf.get_variable('W{}'.format(i), [layer_config[i-1], layer_config[i]], initializer=tf.contrib.layers.xavier_initializer()),
                         'b': tf.Variable(tf.truncated_normal([layer_config[i]], 0, 0.1)), 'M':tf.Variable(tf.truncated_normal(shape=[layer_config[i]],mean=0.5,stddev=0.01, name='Ad_mat')) }
            if i != len(layer_config)-1:
                param['M'].append(new_layer['M']) 
            param['param_W'].append(new_layer['W'])
            param['param_b'].append(new_layer['b'])
            
            l = tf.add(tf.matmul(x if i == 1 else y[i-2], new_layer['W']), new_layer['b'])
              
            l = tf.nn.relu(l) if i != len(layer_config)-1 else l
            
            if sparse == True and i != len(layer_config)-1:
                param['binary_M'].append(binarize(new_layer['M']))
                l = tf.multiply(l, param['binary_M'][-1])
            
            with tf.name_scope(name):
                #l = tf.nn.leaky_relu(l) if i != len(layer_config)-1 else l
                
                if dropout:
                    l = tf.nn.dropout(l, keep_prob=prob) if i != len(layer_config)-1 else l
                #l = tf.layers.batch_normalization(l) if i != len(layer_config)-1 else l
            y[i-1] = l
    
    lastlayer = len(y)-1
    return y[lastlayer] 


def Multi_layer_CNN(x, cnn_config, filter_size, param, sparse_kernel=False, name="CNN"):
    
    y = {}
    binary_WM = []
    with tf.name_scope(name):
        for i in range(1, len(cnn_config)):
            shape = [filter_size, filter_size, cnn_config[i-1], cnn_config[i]]
            
            new_layer = {'WC': tf.get_variable('WC{}'.format(i), shape, initializer=tf.contrib.layers.xavier_initializer()),
                         'bC': tf.Variable(tf.truncated_normal([cnn_config[i]], 0, 0.1)), 'MC': tf.Variable(tf.truncated_normal(shape=[cnn_config[i]],mean=0.5,stddev=0.01))}
            param['MC'].append(new_layer['MC'])
            param['param_WC'].append(new_layer['WC'])
            param['param_bC'].append(new_layer['bC'])
            
            l = tf.nn.conv2d(x if i == 1 else y[i-2], filter=new_layer['WC'], strides=[1, 1, 1, 1], padding='VALID') + new_layer['bC']
            print(l.shape)
            l = tf.nn.relu(l) 
            l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(l.shape)
            if sparse_kernel == True:
                param['binary_MC'].append(binarize_cnn(new_layer['MC']))
                l = tf.multiply(l, param['binary_MC'][-1])
            
            #l = tf.layers.batch_normalization(l) 
                           
            y[i-1] = l

    lastlayer = len(y)-1
    return y[lastlayer], y #[x['W_M'] for _, x in layers.items()]


z = 4
# Amplifier for mask hyper parameters
k = 0.1
# Hyper Param
# Regular learning rate
lr = 1e-3  #  -3 
# Sparsify learning rate
lr_sparse = 1.5e-2 # 1e-1
# Sparsify hyper parameters
lamda_sparse_fc = 0.1 # 0.4
lamda_sparse_cnn = 0.01 # 0.2
lamda_list = [0.1*lamda_sparse_cnn, lamda_sparse_cnn]
# Common L1 norm hyper parameters
lamda_l1 = 1e-5 # 1e-5, 1e-6, 1e-7
# Common L2 norm hyper parameters
lamda_l2 = 5e-5
# Negative penalty should smaller than positive penalty
n_ratio = 0.001
batch_size = 200

tf.reset_default_graph()
# Dict of all parameters
param = {'param_W':[], 'param_b':[], 'param_WC':[], 'param_bC':[], 'M':[],'binary_M':[], 'MC':[],'binary_MC':[]}
# Input and Output placeholders
x = tf.placeholder(tf.float32, shape=[None,image_size,image_size,ch], name='input_x')
y_true = tf.placeholder(tf.float32, [None,n_output], name='y_true')
# Keep probability for dropout 
prob = tf.placeholder_with_default(0.8, shape=())
# Scheduling placeholder 
regu_rate = tf.placeholder(tf.float32, shape=[])

# Network Flow
y, cnn_layers = Multi_layer_CNN(x, n_layer_cnn, kernel_size, param, sparse_kernel=True)
print(y.shape)
yy = tf.layers.flatten(y)
y = Multi_layer_fc(yy, n_layer_fc, param, sparse=True, dropout=True)

# Calculate the total number of parameters in the model and the percentage of parameters
bin_all_fc_list = list(map(tf.reduce_sum, param['binary_M']))
bin_all_fc = sum(bin_all_fc_list)
bin_all_cnn_list =  list(map(tf.reduce_sum, param['binary_MC']))
bin_all_cnn = tf.cast(sum(bin_all_cnn_list), tf.float32)    # convert tf.int to tf.float32 

n_total_cnn = sum(n_layer_cnn) - n_layer_cnn[0]
#n_total_cnn_drop = sum([x*n_layer_cnn[i+1] for i, in enumerate(n_layer_cnn) if i<len(n_layer_cnn)-1])
n_total_fc = sum(n_layer_fc) - n_layer_fc[-1]
n_total_param = n_total_cnn + n_total_fc 

bin_percent_fc = bin_all_fc/n_total_fc
bin_percent_cnn = bin_all_cnn/n_total_cnn
bin_percent_cnn1 = tf.cast(bin_all_cnn_list[0], tf.float32) /20
bin_percent_cnn2 = tf.cast(bin_all_cnn_list[1], tf.float32)/50


#bin_percent_cnn_drop = bin_all_cnn/n_total_cnn_drop
bin_percent = (bin_all_fc+bin_all_cnn)/(n_total_fc+n_total_cnn)

pos_penalty = lamda_sparse_fc*bin_percent_fc + z*lamda_sparse_cnn*bin_percent_cnn1 + lamda_sparse_cnn*bin_percent_cnn2
# Cross Entropy Loss
# L = L + ||w||1 + ||w||2 + ||bin_cnn|| + ||bin_fc|| + ||m||2
cross_entropy_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y)) 
cross_entropy_acc += pos_penalty
#cross_entropy_sp =cross_entropy_acc + lamda_fc*bin_percent_fc + sum([k*i/n for k,i,n in zip(lamda_list, bin_all_cnn_list, n_total_cnn_list)])
        
# Add 2-norm for w
cross_entropy_acc += lamda_l2*sum([tf.nn.l2_loss(param['param_W'][i]) for i in range(len(param['param_W']))])
cross_entropy_acc += lamda_l2*sum([tf.nn.l2_loss(param['param_WC'][i]) for i in range(len(param['param_WC']))])
# Add 1-norm for w
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lamda_l1, scope=None)
cross_entropy_acc += tf.contrib.layers.apply_regularization(l1_regularizer, param['param_W'])
cross_entropy_acc += tf.contrib.layers.apply_regularization(l1_regularizer, param['param_WC'])

optimizer_acc = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_acc, var_list=[param['param_W'], param['param_WC'],param['param_bC'], param['param_bC']]) 

optimizer_sp = tf.train.AdamOptimizer(learning_rate=regu_rate).minimize(cross_entropy_acc, var_list=[param['M'], param['MC']])

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
    saver.restore(sess, "../MNIST_lenet_neuron.ckpt")
    for step in range(200000):
        batch_x , batch_y = mnist.train.next_batch(batch_size)
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        batch_x = np.reshape(batch_x, [-1, image_size, image_size, ch])
        test_images = np.reshape(mnist.test.images, [-1, image_size, image_size, ch])
        
        if step % 1 == 0:
            _, loss_acc, par,par2 = sess.run([optimizer_acc, cross_entropy_acc, param['MC'],param['M']] ,feed_dict={x: batch_x, y_true: batch_y})   
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
            matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
            evaluate = tf.reduce_mean(tf.cast(matches,tf.float32))
            accuracy_sparse_before.append(sess.run(evaluate,feed_dict={x:batch_x, y_true:batch_y, prob:1}))
            accuracy_sparse.append(sess.run(evaluate,feed_dict={x:test_images, y_true:test_labels,prob:1}))
            
            if step < 0:
                print("step = {0:5d}, l_acc = {1:2.5f}, err = {2:2.2f}, w_0/w = {3:2.2f}"
                      .format(step, loss_acc, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100))
            else:
                print("step = {0:5d}, l_acc = {1:2.5f}, l_sp = {2:2.5f}, err_train = {6:2.2f}, err_test = {3:2.2f}, w_0/w = {4:2.2f}, X = {5:2.0f}"
                      .format(step, loss_acc, loss_sp, (1-accuracy_sparse[-1])*100, sparse_sparse[-1]*100, 1/sparse_sparse[-1],(1-accuracy_sparse_before[-1])*100 ))
                cnn_per1, cnn_per2, fc_per = sess.run([bin_percent_cnn1,bin_percent_cnn2, bin_percent_fc] ,feed_dict={x: batch_x, y_true: batch_y})
                print("cnn1={0:2.2f}%, cnn2={1:2.2f}%, fc={2:2.2f}%".format(cnn_per1*100,cnn_per2*100, fc_per*100))
print(time.time() - init_time)