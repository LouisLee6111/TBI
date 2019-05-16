import numpy as np 
import os
import file_loader as fl
import artifact_removal as ar
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
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score, accuracy_score

def get_original_autoencoder(input_shape, filters=192, kernel_size=5, depth=4, pool_size=2, concat_axis=2):
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

def get_original_encoder(input_shape, filters=192, kernel_size=5, depth=4, pool_size=2, concat_axis=2):
  inputs = Input((input_shape,1))
  encoded=get_encoder_part(inputs, input_shape, filters,  kernel_size, depth, pool_size, concat_axis )
  encoder = Model(inputs, encoded)
  adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5)
  encoder.compile(optimizer=adam, loss="binary_crossentropy")
  return encoder

def get_encoder_part(inputs,input_shape,filters=192, kernel_size=5, depth=4, pool_size=2, concat_axis=2):
  for i in range(depth):
    if i==0:
      conc=inputs

    conv = Conv1D(filters=filters, kernel_size=1, activation='relu', padding='same', kernel_initializer= 'he_normal',  name='en_conv1_' + str(i+1))(conc)
    conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer= 'he_normal',  name='en_conv2_' + str(i+1))(conv)
    pool = MaxPooling1D(pool_size, padding='same', name="pool_"+str(i+1))(conv)

    p=int(pow(pool_size,i+1))
    f=filters//p
    a=input_shape//p
    input_conv = Conv1D(filters=f, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer= 'he_normal',  name='en_shortcut'+ str(i+1))(inputs)
    shortcut = Reshape((a,filters))(input_conv)
    conc=concatenate([pool, shortcut], name='en_conc_' + str(i+1), axis=concat_axis)
   
  encoded = Conv1D(filters=filters,kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer = 'he_normal',name = 'encoded_cov1')(conc)
  encoded = Conv1D(1,kernel_size=1,activation='sigmoid',padding='same',kernel_initializer = 'he_normal',name = 'encoded_cov2')(encoded)

  return encoded

def artifact_labeling(data, win_size=128):
  n_abp = ar.norm_unit(data.abp)
  i = 0
  seg_abp = []
  while i + win_size < len(n_abp):
    seg_abp.append(n_abp[i:i+win_size])
    i += win_size
  i = 0
  y = np.zeros(len(seg_abp))
  for v in data.abp_artifact:
    str_idx = data.ldtime.index(v[0])
    end_idx = data.ldtime.index(v[1])
    now_pulse_start = 0
    now_pulse_end = len(seg_abp[0])
    while True:
      if i+1 >= len(seg_abp): break;
      if str_idx > now_pulse_start and str_idx < now_pulse_end:
        break
      i += 1
      now_pulse_start = now_pulse_end + 1
      now_pulse_end += len(seg_abp[i])
    str_i = i
    while True:
      if i+1 >= len(seg_abp): break;
      if end_idx > now_pulse_start and end_idx < now_pulse_end:
        break
      i += 1
      now_pulse_start = now_pulse_end + 1
      now_pulse_end += len(seg_abp[i])
    end_i = i

    for j in range(str_i, end_i):
      y[j] = 1
  return seg_abp, y
    
def class_data_gen():
  os.chdir('E:/Richard/TBI/com_ar_dat_icp/')
  cur_files = os.listdir()
  file_path = 'E:/Richard/TBI/ar_ori_data/'
  files = os.listdir(file_path)
  total_abp = []; total_icp = [];
  xs = []; ys = [];
  for file in files:
    if file.split('_')[0] +'.npz' in cur_files: continue
    print(file)
    data = fl.read_csv(file_path+file)
    data.load_artifact_info('E:/Richard/TBI/ar_to_csv/icp/', file.split('_')[0])
    x, y = artifact_labeling(data)
    x = np.array(x)
    y = np.array(y)
    np.savez_compressed(file.split('_')[0], x=x, y=y)

def label_translator(y, mode=1):
  # mode 1: 1d-arr to nd-arr
  # mode 2: nd-arr to 1d-arr
  rev_y = []
  if mode == 1:
    for v in y:
      if v == 0: rev_y.append([0, 1])
      else: rev_y.append([1, 0])
  else:
    for v in y:
      if v == [0, 1]: rev_y.append(0)
      else: rev_y.append(1)
  return np.array(rev_y)

def ae_training():
  os.chdir('E:/Richard/TBI/com_ar_dat_icp/')
  files = os.listdir()
  data_arr = []
  for file in files:
    if file.split('.')[-1] != 'npz':
      continue
    data = np.load(file)
    data_arr.append(data)
  for i in range(len(data_arr)):
    test_x = data_arr[i]['x']
    test_y = label_translator(data_arr[i]['y'])
    train_x = []; train_y = [];
    for j in range(len(data_arr)):
      if i == j: continue
      train_x.extend(data_arr[j]['x'])
      train_y.extend(label_translator(data_arr[j]['y']))
    win_size = 128
    ae_depth = 3
    ae = get_original_autoencoder(win_size, depth=ae_depth)
    print(ae.summary())
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)
    ae.fit(train_x, train_x, epochs=5)
    ae_json = ae.to_json()
    with open('net/' + str(i) + '.net', 'w') as json_file:
      json_file.write(ae_json)
    ae.save_weights('weight/' + str(i) + '.wgt', overwrite='True')

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def percentage_RMS_Difference(y_true, y_pred):
  return K.sqrt( K.sum(K.square(y_pred - y_true)) / K.sum(K.square(y_true)) ) * 100.0

def get_ad_classifier_blstm(input_shape, output_shape, filters=24):

  inputs = Input((input_shape,1))
  lstm = Bidirectional( LSTM(filters, activation='tanh', return_sequences=True, name='1_blstm'))(inputs)
  drop = Dropout(0.15, name='2_dropout')(lstm)
  lstm = Bidirectional( LSTM(filters, activation='tanh', return_sequences=True, name='3_blstm'))(drop)
  drop = Dropout(0.15, name='4_dropout')(lstm)
  lstm = Bidirectional( LSTM(filters, activation='tanh', return_sequences=True, name='5_blstm'))(drop)
  drop = Dropout(0.15, name='6_dropout')(lstm)
  lstm = Bidirectional( LSTM(filters, activation='tanh', name='7_blstm'))(drop)
  dens = Dense(output_shape, activation='softmax', name='8_dens')(lstm)

  model = Model(inputs, dens)

  adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5)

  model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy'])
  return model


def artifact_classification():
  os.chdir('E:/Richard/TBI/com_ar_dat_icp/')
  win_size = 128
  ae_depth = 3
  files = os.listdir('weight')
  models = []
  for file in files:
    model = get_original_encoder(win_size, depth=ae_depth)
    model.load_weights('weight/'+file, by_name=True)
    models.append(model)

  files = os.listdir()
  data_arr = []
  for file in files:
    if file.split('.')[-1] != 'npz':
      continue
    data = np.load(file)
    data_arr.append(data)

  for i in range(1, len(data_arr)):
    test_x = data_arr[i]['x']
    test_y = label_translator(data_arr[i]['y'])
    train_x = []; train_y = [];
    for j in range(len(data_arr)):
      if i == j: continue
      train_x.extend(data_arr[j]['x'])
      train_y.extend(label_translator(data_arr[j]['y']))
    
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)
    enc_train_x = models[i].predict(train_x)
    enc_test_x = models[i].predict(test_x)

    enc_train_x = np.array(enc_train_x)
    enc_test_x = np.array(enc_test_x)

    enc_train_x = np.reshape(enc_train_x, (enc_train_x.shape[0], enc_train_x.shape[1], 1))
    enc_test_x = np.reshape(enc_test_x, (enc_test_x.shape[0], enc_test_x.shape[1], 1))

    enc_train_x = enc_train_x.astype('float32')
    enc_test_x = enc_test_x.astype('float32')

    train_y = np.array(train_y).astype('float32')
    test_y = np.array(test_y).astype('float32')

    clf_model = get_ad_classifier_blstm(16, 2)
    print(clf_model.summary())
    clf_model.fit(enc_train_x, np.array(train_y), validation_data=(enc_test_x, np.array(test_y)), epochs=1)
    clf_model.save('classifier/net_1/' + str(i) + '.net')


def performance():
  from sklearn.metrics import confusion_matrix
  os.chdir('E:/Richard/TBI/com_ar_dat/')
  win_size = 128
  ae_depth = 3
  files = os.listdir('weight')
  models = []
  for file in files:
    model = get_original_encoder(win_size, depth=ae_depth)
    model.load_weights('weight/'+file, by_name=True)
    models.append(model)

  files = os.listdir()
  data_arr = []
  for file in files:
    if file.split('.')[-1] != 'npz':
      continue
    data = np.load(file)
    data_arr.append(data)
  for i in range(len(data_arr)):
    test_x = data_arr[i]['x']
    test_y = label_translator(data_arr[i]['y'])
    train_x = []; train_y = [];
    for j in range(len(data_arr)):
      if i == j: continue
      train_x.extend(data_arr[j]['x'])
      train_y.extend(label_translator(data_arr[j]['y']))
    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)
    enc_train_x = models[i].predict(train_x)
    enc_test_x = models[i].predict(test_x)
    enc_train_x = np.array(enc_train_x)
    enc_test_x = np.array(enc_test_x)
    enc_train_x = np.reshape(enc_train_x, (enc_train_x.shape[0], enc_train_x.shape[1], 1))
    enc_test_x = np.reshape(enc_test_x, (enc_test_x.shape[0], enc_test_x.shape[1], 1))
    enc_train_x = enc_train_x.astype('float32')
    enc_test_x = enc_test_x.astype('float32')
    
    clf_model = load_model('classifier/net_1/' + str(i) + '.net')
    pred = clf_model.predict(enc_test_x)

    cm = confusion_matrix(test_y.argmax(axis=1), pred.argmax(axis=1))
    acc = accuracy_score(pred.argmax(axis=1), test_y.argmax(axis=1))
    kap = cohen_kappa_score(pred.argmax(axis=1), test_y.argmax(axis=1))
    sen = cm[0, 0]/(cm[0, 0] + cm[0, 1])
    spe = cm[1, 1]/(cm[1, 1] + cm[1, 0])
    pen = open('tbi_abp_ar_res.csv', 'a')
    pen.write(str(acc) + ',' + str(kap) + ',' + str(sen) + ',' + str(spe) + '\n')
    print('abc')



if __name__ == '__main__':
  #class_data_gen()
  #ae_training()
  #artifact_classification()
  performance()
  """
  win_size = 128
  ae_depth = 3
  get_original_autoencoder(win_size, depth=ae_depth)

  """