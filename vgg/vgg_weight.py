import os
import argparse
import time
from numpy import prod
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

from models import build_vgg_weight_model
from layers import binarize
from utils import *


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--n_output', type=int, default=10, help='number of outputs')
parser.add_argument('--lr_sparse', type=float, default=1.5e-2, help='auxiliary parameter learning rate')
parser.add_argument('--lr', type=float, default=1e-5, help='weight learning rate')
parser.add_argument('--lamda_sparse_fc', type=float, default=0.05, help='fully connect layers sparse regularization')
parser.add_argument('--lamda_sparse_cnn', type=float, default=1, help='cnn layers sparse regularization')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--lamda_l1', type=float, default=1e-6, help='L1 regularization')
parser.add_argument('--lamda_l2', type=float, default=5e-4, help='L2 regularization')
parser.add_argument('--n_ratio', type=float, default=1e-3, help='slope for recoverability')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--prob', type=float, default=0.3, help='dropout rate')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

IMAGE_SIZE = 32
CH = 3
CIFAR_CLASSES = 10
HIDDEN = 512


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf.reset_default_graph()

    vgg = build_vgg_weight_model(args.lamda_l1, args.lamda_l2, args.prob)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, CH], name='input_x')
    y_true = tf.placeholder(tf.float32, [None, CIFAR_CLASSES], name='y_true')
    regu_rate = tf.placeholder(tf.float32, shape=[])
    # Read VGG Model
    y = vgg(x)

    # Get Mask Variables
    all_mask_variables = [v for v in tf.global_variables() if "mask" in v.name and "Adam" not in v.name]
    cnn_mask_variables = [v for v in all_mask_variables if
                          "kernel" not in v.name and "bias" not in v.name and "conv" in v.name]
    fc_mask_variables = [v for v in all_mask_variables if
                         "kernel" not in v.name and "bias" not in v.name and "conv" not in v.name]

    all_binary_layers = [t for tensor in tf.get_default_graph().get_operations() for t in tensor.values()
                         if"binary" in tensor.name]
    cnn_mask_layers = [v for v in all_binary_layers if "conv" in v.name][:13]
    cnn_mask_shape = [v.get_shape().as_list() for v in cnn_mask_layers]
    fc_mask_layers = [v for v in all_binary_layers if "dense" in v.name][:2]


    bin_all_cnn_list = list(map(tf.reduce_sum, cnn_mask_layers))
    bin_all_cnn = tf.cast(sum(bin_all_cnn_list), tf.float32)
    n_total_cnn = sum(map(prod, cnn_mask_shape))
    bin_percent_cnn = bin_all_cnn / n_total_cnn

    bin_all_fc_list = list(map(tf.reduce_sum, fc_mask_layers))
    bin_all_fc = tf.cast(sum(bin_all_fc_list), tf.float32)
    n_total_fc = HIDDEN * HIDDEN + HIDDEN * CIFAR_CLASSES  # sum(map(sum, [fc_mask_shape]))
    bin_percent_fc = bin_all_fc / n_total_fc
    bin_percent = (bin_all_fc + bin_all_cnn) / (n_total_fc + n_total_cnn)

    # Collect all weights
    all_weights = [var for var in tf.global_variables() if "kernel" in var.name and "Adam" not in var.name]
    all_bias = [var for var in tf.global_variables() if "bias" in var.name and "Adam" not in var.name]
    pos_penalty = args.lamda_sparse_fc * bin_percent_fc + args.lamda_sparse_cnn * bin_percent_cnn
    cross_entropy_acc_ori = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
    cross_entropy_acc = cross_entropy_acc_ori + pos_penalty

    # Adding neg penality
    nRegu = [binarize(tf.negative(x)) for x in fc_mask_variables]
    nRegu_cnn = [binarize(tf.negative(x)) for x in cnn_mask_variables]
    nbin_all_fc = sum(list(map(tf.reduce_sum, nRegu)))
    nbin_all_cnn = sum(list(map(tf.reduce_sum, nRegu_cnn)))
    neg_penalty = args.lamda_sparse_fc * args.n_ratio * (nbin_all_fc / n_total_fc) + \
                  args.lamda_sparse_cnn * args.n_ratio * (nbin_all_cnn / n_total_cnn)
    cross_entropy_acc += neg_penalty

    optimizer1 = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-8)
    gradients1, vriables1 = zip(*optimizer1.compute_gradients(cross_entropy_acc_ori, var_list=[all_weights, all_bias]))
    # abs_invers_var = [tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape),
    #                   tf.multiply(tf.abs(tf.reciprocal(x)), tf.sqrt(tf.abs(tf.reciprocal(x))))) for x in vriables1]
    abs_invers_var = [tf.where(tf.less(tf.abs(x), 1e-15), tf.constant(0., shape=x.shape),
                               tf.abs(tf.reciprocal(x))) for x in vriables1]

    optimizer2 = tf.train.AdamOptimizer(learning_rate=regu_rate, epsilon=1e-8)
    gradients2, vriables2 = zip(*optimizer2.compute_gradients(cross_entropy_acc,
                                                              var_list=[cnn_mask_variables, fc_mask_variables]))
    boost_gradient2 = [tf.multiply(x, y) for x, y in zip(gradients2, abs_invers_var[:15])]
    optimizer_sp = optimizer2.apply_gradients(zip(boost_gradient2, vriables2))

    optimizer_1 = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-8)
    gradients_1, vriables_1 = zip(*optimizer_1.compute_gradients(cross_entropy_acc, var_list=[all_weights, all_bias]))
    optimizer_acc = optimizer_1.apply_gradients(zip(gradients_1, vriables_1))

    datagen = ImageDataGenerator()

    init = tf.global_variables_initializer()
    accuracy_sparse_before = [0]
    accuracy_sparse = [0]
    sparse_sparse = [0]
    percent = 1

    with tf.Session() as sess:
        sess.run(init)
        vgg.load_weights("vgg_model_kernel_init.h5")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train, CIFAR_CLASSES)
        y_test = keras.utils.to_categorical(y_test, CIFAR_CLASSES)
        train_val = 25000

        x_train_val, y_train_val = x_train[train_val:], y_train[train_val:]
        x_train, y_train = x_train[:train_val], y_train[:train_val]

        train_gen = datagen.flow(x=x_train, y=y_train, batch_size=args.batch_size)
        test_gen = datagen.flow(x=x_test, y=y_test, batch_size=2000)
        train_val_gen = datagen.flow(x=x_train_val, y=y_train_val, batch_size=args.batch_size)

        for step in range(int(args.epochs * 50000 / args.batch_size)):
            batch_x, batch_y = train_gen.next()
            batch_x_val, batch_y_val = train_val_gen.next()

            _, percent, loss_sp = sess.run([optimizer_sp, bin_percent, cross_entropy_acc],
                                           feed_dict={x: batch_x_val, y_true: batch_y_val,
                                                      regu_rate: args.lr_sparse, K.learning_phase(): 0})

            _, loss_acc, par = sess.run([optimizer_acc, cross_entropy_acc, cnn_mask_variables],
                                        feed_dict={x: batch_x, y_true: batch_y, K.learning_phase(): 0})

            if step % 500 == 0:
                sparse_sparse.append(percent)
                batch_test_x, batch_test_y = test_gen.next()
                matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
                evaluate = tf.reduce_mean(tf.cast(matches, tf.float32))
                accuracy_sparse_before.append(
                    sess.run(evaluate, feed_dict={x: batch_x, y_true: batch_y, K.learning_phase(): 0}))
                accuracy_sparse.append(
                    sess.run(evaluate, feed_dict={x: batch_test_x, y_true: batch_test_y, K.learning_phase(): 0}))

                print("step = {0:5d}, loss_acc = {1:2.5f}, loss_sp = {2:2.5f}, err_train = {6:2.2f}, err_test = {3:2.2f}, "
                      "w_0/w = {4:2.2f}, X = {5:2.0f}"
                      .format(step, loss_acc, loss_sp, (1 - accuracy_sparse[-1]) * 100, sparse_sparse[-1] * 100,
                              1 / sparse_sparse[-1], (1 - accuracy_sparse_before[-1]) * 100))


if __name__ == '__main__':
  main()