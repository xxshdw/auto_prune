from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras import regularizers
from layers import MaskConv, MaskDense, MaskLayer
#import tensorflow.keras.layers.SpatialDropout2D as Dropout

def build_vgg_weight_model(lamda_l1, lamda_l2, prob):
    model = Sequential()

    model.add(MaskConv(64, (3, 3), input_shape=[32, 32, 3], padding='same',
                       kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskConv(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(prob))

    model.add(Flatten())
    model.add(MaskDense(512, kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(MaskDense(10, kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('softmax'))

    return model


def build_vgg_neuron_model(lamda_l1, lamda_l2, prob):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=[32, 32, 3], kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn11"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn12"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn21"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn22"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn31"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn32"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn33"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn41"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn42"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn43"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn51"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn52"))
    model.add(BatchNormalization())
    model.add(Dropout(prob))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("cnn53"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(prob))

    model.add(Flatten())
    model.add(MaskLayer("fc0"))
    model.add(Dense(512, kernel_regularizer=regularizers.L1L2(l1=lamda_l1, l2=lamda_l2)))
    model.add(Activation('relu'))
    model.add(MaskLayer("fc1"))
    model.add(BatchNormalization())

    model.add(Dropout(prob))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model