import numpy as np 
import os

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import CSVLogger
from keras.regularizers import l2

def get_autoencoder(input_shape, filter=192, kernel_size=5, depth=4, pool_size=2, concat_axis=2):
  inputs = Input((input_shape,1)) 
  encoded=get_encoder_part(inputs, input_shape, filters,  kernel_size, depth, pool_size, concat_axis )
  a=input_shape/(pow(pool_size, depth ))
  for i in range(depth):
    if i==0:
      conc = encoded
    conv = Conv1D(filters=filters, kernel_size=1,activation='relu',padding='same',kernel_initializer = 'he_normal',   name='de_conv1_' + str(i+1))(conc)
    conv = Conv1D(filters=filters, kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer = 'he_normal',   name='de_conv2_' + str(i+1))(conv)
    us = UpSampling1D(pool_size, name='upsm_' + str(i+1))(conv)

    p=int(pow(pool_size,i+1))
    a=a*pool_size
    f=int(filters*p)
    
    shortcut = Conv1D(filters=f, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer= 'he_normal',  name='de_shortcut'+ str(i+1))(encoded)
    shortcut = Reshape((a,filters))(shortcut)
    conc=concatenate([us, shortcut], name='de_conc_' + str(i+1), axis=concat_axis)

  decoded = Conv1D(filters=filters,kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer = 'he_normal',name = 'decoded_cov1')(conc)
  decoded = Conv1D(1,kernel_size=1,activation='sigmoid',padding='same',kernel_initializer = 'he_normal',name = 'decoded_cov2')(decoded)

  autoencoder = Model(inputs, decoded)
  adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5)
  autoencoder.compile(optimizer=adam, loss="mean_squared_error",  metrics=[root_mean_squared_error, percentage_RMS_Difference])
  return autoencoder


