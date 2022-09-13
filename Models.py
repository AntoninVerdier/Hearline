import os
import time
import tensorflow as tf

import tensorflow.keras
import numpy as np

from tcn import TCN

import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, DepthwiseConv2D, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout, Conv1D, UpSampling1D, MaxPooling1D, AveragePooling1D

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

import tensorflow.experimental.numpy as tnp
import tensorflow_probability as tfp

# from AE import Sampling
# from AE import VAE

from sklearn import preprocessing as p


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import matplotlib.pyplot as plt
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

class DenseMax(Layer):
    """
        Custom kera slayer 
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 max_n=100,
                 lambertian=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.max_n = max_n
        self.lambertian = lambertian
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]


        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)

        # Maybe useful to store filter in object to study evolution after
        sorted_output = tnp.sort(output)
        threshold_for_each_batch = sorted_output[:, -(self.max_n + 1)]
        filter_bool = tnp.transpose(tnp.greater(tnp.transpose(output), threshold_for_each_batch))
        output = tnp.multiply(output, filter_bool)

        if self.lambertian:
            output = output.reshape(-1, 10, 10)
            output = output.reshape(-1, 100)

        return output

    def get_config(self):
        base_config = super(DenseMax, self).get_config()
        base_config['max_n'] = self.max_n
 

        return base_config

class Autoencoder():
    # This class should return the required autoencoder architecture
    def __init__(self, model, input_shape, latent_dim, dataset_type='log', max_n=None, toeplitz_spec=None, toeplitz_true=None):
        self.model = model
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dataset_type = dataset_type

    def get_model(self):
        if self.model == 'tcn_ae':
            return self.__tcn_ae()

    def __tcn_ae(self):
        dilations = (1, 2, 4, 16)
        nb_filters = 20
        kernel_size = 20
        nb_stacks = 1
        padding = 'same'
        dropout_rate = 0.00
        filters_conv1d = 32
        activation_conv1d = 'linear'
        latent_sample_rate = 64
        pooler = AveragePooling1D
        lr = 0.001
        conv_kernel_init = 'glorot_normal'
        loss = 'logcosh'
        sampling_factor = latent_sample_rate

        print(self.input_shape)
        
        
        tensorflow.keras.backend.clear_session()
        inp = Input((self.input_shape[1], 1))

        # Put signal through TCN. Output-shape: (batch,sequence length, nb_filters)
        tcn_enc = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations, 
                      padding=padding, use_skip_connections=True, dropout_rate=dropout_rate, return_sequences=True,
                      kernel_initializer=conv_kernel_init, name='tcn-enc')(inp)

        # Now, adjust the number of channels...
        enc_flat = Conv1D(filters=filters_conv1d, kernel_size=1, activation=activation_conv1d, padding=padding)(tcn_enc)

        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        enc_pooled = pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        
        # If you want, maybe put the pooled values through a non-linear Activation
        encoded = LeakyReLU(alpha=0.3, name='encoded')(enc_pooled)

        
        dec_upsample = UpSampling1D(size=sampling_factor)(encoded)

        dec_reconstructed = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations, 
                                padding=padding, use_skip_connections=True, dropout_rate=dropout_rate, return_sequences=True,
                                kernel_initializer=conv_kernel_init, name='tcn-dec')(dec_upsample)

        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        output = Dense(1, activation='linear', dtype='float32')(dec_reconstructed)        


        autoencoder = Model(inputs=inp, outputs=[output])

        adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        
        autoencoder.compile(loss=loss, optimizer=adam, metrics=[loss])
        
        print(autoencoder.summary())

        return autoencoder