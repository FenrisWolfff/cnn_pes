#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyvibdmc as dmc
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *
import tensorflow as tf
from tensorflow.keras.layers import Dense
from try_descriptors import *

# In[3]:
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

def vectorize(clmat):
    tril_map = 1 - np.tril(np.ones((6,6)))
    cleanmat = np.multiply(clmat, tril_map).flatten()
    return cleanmat[cleanmat != 0]

def normalize(vec, mx, mn):
    return 2*(vec-mn)/(mx-mn) - 1


def ch5_pot(cds):
    model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(15,)),
        Dense(128, activation=tf.nn.swish),
        Dense(128, activation=tf.nn.swish),
        Dense(128, activation=tf.nn.swish),
        Dense(1, activation=tf.nn.swish)
    ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                     loss='mse',
                     metrics=[])
    model.load_weights('ch5_3x128.h5')
    
    cds= coulomb_it(cds)
    cds = np.array([vectorize(mat) for mat in cds])
    cds = normalize(cds, 6, 0)
    pots_wn = (10**model.predict(cds)).flatten()
    return Constants.convert(pots_wn, 'wavenumbers', to_AU=True)

# In[ ]:




