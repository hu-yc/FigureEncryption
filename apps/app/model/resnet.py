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


if __name__ == "__main__":
    import numpy as np

    model = get_resnet()
    model.load_weights('../conf/d_A_epoch100.h5', skip_mismatch=True, by_name=True)
    raw_seed = np.random.randint(0, 20000001, size=(1, 256, 256, 3))
    random_seed = np.array(raw_seed, dtype=np.float32) / 10000000 - 1
    latent_fake = model.predict(random_seed)
    with open('../../test/sample.npy', 'wb') as f:
        np.save(f, latent_fake)
