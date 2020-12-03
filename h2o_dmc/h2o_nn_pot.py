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


# In[3]:



# In[4]:


def internals_h2o(cds):
    analyzer = AnalyzeWfn(cds)
    bl1 = analyzer.bond_length(0,2)
    bl2 = analyzer.bond_length(1,2)
    theta = analyzer.bond_angle(0,2,1)
    return np.array((bl1,bl2,theta)).T

def h2o_pot(cds):
    model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(3,)),
        Dense(64, activation=tf.nn.swish),
        Dense(64, activation=tf.nn.swish),
        Dense(64, activation=tf.nn.swish),
        Dense(1, activation='linear')
    ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                     loss='mse',
                     metrics=[])
    model.load_weights('h2o_3x64.h5')
    internal_cds = internals_h2o(cds)
    norm_cds = 2*(internal_cds)/3-1
    pots_wn = (10**model.predict(norm_cds)).flatten()
    return Constants.convert(pots_wn, 'wavenumbers', to_AU=True)

# In[ ]:




