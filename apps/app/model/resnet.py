import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


K.set_floatx('float64')


def get_resnet():
    img_input = Input(shape=(256, 256, 3))
    x = Flatten()(img_input)
    x = Dense(128, activation="sigmoid", trainable=True, name='densea', kernel_initializer='random_uniform',
              bias_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None))(x)
    x = Dense(256 * 256 * 1, activation="sigmoid", trainable=True, name='denseb', kernel_initializer='random_uniform',
              bias_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None))(x)
    x = Reshape((256, 256, 1))(x)
    x = Conv2D(64, (4, 4), padding='same', trainable=False, name='conv1', kernel_initializer='random_uniform',
               bias_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None))(x)
    x = InstanceNormalization(axis=3, name='ins1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (4, 4), padding='same', trainable=False, name='conv2', kernel_initializer='random_uniform',
               bias_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None))(x)
    x = InstanceNormalization(axis=3, name='ins2')(x)
    x = Activation('relu')(x)
    x = Conv2D(3, (1, 1), trainable=False, name='conv3', kernel_initializer='random_uniform',
               bias_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None))(x)
    x = Activation('tanh')(x)
    x = Lambda(lambda a: (tf.floor((tf.abs(a) - tf.floor(tf.abs(a))) * 1000000000000000) % 256))(x)
    model = Model(img_input, x)
    return model

