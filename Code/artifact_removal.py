import peak_detection as pd
from scipy.interpolate import CubicSpline
import signals as sn
import file_loader as fl
import numpy as np
import matplotlib.pyplot as plt

def onset_dividing(data, mode='abp'):
  if mode == 'abp': ons = data.abp_ons; val = data.abp;
  else: ons = data.icp_ons; val = data.icp;
  od = []
  for i in range(len(ons) - 1):
    od.append(val[ons[i]:ons[i+1]])
  return od

def interpolation(onset_divided_data, int_num=64):
  id = []
  for op in onset_divided_data: # one pulse
    cs = CubicSpline(np.linspace(0, int_num, len(op)), op)
    #plt.subplot(2, 1, 1)
    #plt.plot(np.linspace(0, int_num, len(op)), op, 'b-')
    #plt.subplot(2, 1, 2)
    y = cs(np.linspace(0, int_num, int_num))
    #plt.plot(np.linspace(0, int_num, int_num), y, 'b-')
    #plt.show()
    id.append(y)
  return id

def norm_unit(id):
  id = np.array(id)
  max_val = id.max()
  min_val = id.min()
  if max_val == min_val: max_val += 0.001
  return (id - min_val)/(max_val - min_val)
def normalization(interpolated_data):
  nd = []
  for id in interpolated_data:
    nd.append(norm_unit(id))
  return nd

def sgn_to_img(sgn):
  imgs = []
  for s in sgn:
    dim = len(s)
    img = np.zeros((dim, dim))
    for i in range(dim):
      idx = int(round(s[i] * (dim-1)))
      img[dim-idx-1][i] = 255
      if idx > 0:
        img[dim-idx][i] = 125
      if idx < dim-1:
        img[dim-idx-2][i] = 125
    imgs.append(img)
  return np.array(imgs)

def gen_rep_img(imgs, mode='abp'):
  rimgs = np.array(imgs).astype('float32')/255
  rimgs = np.reshape(rimgs, (len(rimgs), 64, 64, 1))
  path = 'abp_ae.net' if mode=='abp' else 'icp_ae.net'
  from keras.models import load_model
  ae = load_model('ar/net/'+path)
  rimgs = ae.predict(rimgs)
  return rimgs

def show_img(rimg):
  plt.imshow(rimg.reshape(64, 64))
  plt.show()

def artifact_classification(rimgs, mode='abp'):
  path = 'abp_cnn.net' if mode=='abp' else 'icp_cnn.net'
  #path = 'total_abp_1214.net'
  from keras.models import load_model
  cnn = load_model('ar/net/'+path)
  pred = cnn.predict(rimgs)
  return pred.argmax(axis=1)

def artifact_selector(data, pred, mode='abp'):
  if mode == 'abp':
    val = data.abp; ons = data.abp_ons;
  else:
    val = data.icp; ons = data.icp_ons;
  art_x = []; art_y = [];
  for p in range(len(pred)):
    if pred[p] == 0:
      art_x.extend(np.linspace(ons[p], ons[p+1] - 1, ons[p+1]-ons[p]))
      art_y.extend(val[ons[p]:ons[p+1]])
  return np.array(art_x).astype('int'), art_y

def abp_proc(data):
  od = onset_dividing(data, 'abp')
  id = interpolation(od)
  nd = normalization(id)
  imgs = sgn_to_img(nd)
  rimgs = gen_rep_img(imgs)
  pred = artifact_classification(rimgs)
  return pred
  #x, y = artifact_selector(data, pred)
  #plt.plot(np.linspace(0, len(data.abp)-1, len(data.abp)), data.abp, 'b-', x, y, 'r*')
  #plt.show()
  #return x, y

def icp_proc(data):
  od = onset_dividing(data, 'icp')
  id = interpolation(od)
  nd = normalization(id)
  imgs = sgn_to_img(nd)
  rimgs = gen_rep_img(imgs)
  pred = artifact_classification(rimgs, 'icp')
  return pred
  #x, y = artifact_selector(data, pred, 'icp')
  #plt.plot(np.linspace(0, len(data.icp)-1, len(data.icp)), data.icp, 'b-', x, y, 'r*')
  #plt.show()
  #return x, y

def simul_reader():
  sgn = sn.Signal('sim')
  file = open('ar/sim/artifact.csv')
  line = file.readline()
  file.close
  sl = line.split(',')
  from datetime import datetime, timedelta
  import pytz
  now_time = datetime(1899,12,30,0,0,0,tzinfo=pytz.utc)
  for v in sl:
    sgn.abp.append(float(v))
    sgn.icp.append(float(v))
    sgn.dtime.append(now_time)
    now_time += timedelta(milliseconds=8)
  return sgn

def onset_showing():
  data = fl.read_csv('E:/Richard/TBI/testdata/215_1.csv')
  data.gen_mor()
  x, y = icp_proc(data)
  

  #plt.imshow(imgs[10].reshape(64, 64))
  #plt.show()

def autoencoding_cnn(train_x, test_x, net_path, img_dim=64, encoding_dim=32):
  from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
  from keras.models import Model

  input_img = Input(shape=(img_dim, img_dim, 1))
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)
  # at this point the representation is (7, 7, 32)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  r_train_x = np.array(train_x).astype('float32')/255
  r_train_x = np.reshape(r_train_x, (len(r_train_x), img_dim, img_dim, 1))
  r_test_x = np.array(test_x).astype('float32')/255
  r_test_x = np.reshape(r_test_x, (len(r_test_x), img_dim, img_dim, 1))
  #print(autoencoder.summary())
  autoencoder.fit(r_train_x, r_train_x, epochs=100, batch_size=100, shuffle=True)
  decoded_imgs = autoencoder.predict(r_test_x)
  autoencoder.save(net_path)
  return decoded_imgs

def classifier_cnn(ipt_dim):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(ipt_dim, ipt_dim, 1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def load_ar_dat():
  import os
  path = 'ar/dat/ar.csv/'
  abp_files = os.listdir(path + 'abp')
  abp_ar = {}
  for file in abp_files:
    f = open(path + 'abp/' + file)
    abp_ar[file] = []
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
      sl = line.split(',')
      abp_ar[file].append([float(sl[0]), float(sl[1])])
  icp_files = os.listdir(path + 'icp')
  icp_ar = {}
  for file in icp_files:
    f = open(path + 'icp/' + file)
    icp_ar[file] = [] 
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
      sl = line.split(',')
      icp_ar[file].append([float(sl[0]), float(sl[1])])
  return abp_ar, icp_ar

def classifier_train():
  import os


def representative_train():
  import os
  path = 'ar/dat/img(npz)/ori/'
  files = os.listdir(path)
  imgs_abp = []; imgs_icp = [];
  for file in files:
    print(file)
    data = np.load(path + file)
    imgs_abp.extend(list(data['abp']))
    imgs_icp.extend(list(data['icp']))
  imgs_abp = np.array(imgs_abp); imgs_icp = np.array(imgs_icp);  
  rimgs_abp = autoencoding_cnn(imgs_abp, imgs_abp, 'ar/net/gen/abp_ae.net')
  rimgs_icp = autoencoding_cnn(imgs_icp, imgs_icp, 'ar/net/gen/icp_ae.net')
  np.savez_compressed('ar/dat/img(npz)/rep/total', abp=imgs_abp, icp=imgs_icp)
  return rimgs_abp, rimgs_icp


def save_morphological_feature():
  import os, pickle, gzip
  files = os.listdir('ar/dat/ori/')
  for file in files:
    print(file)
    data = fl.read_csv('ar/dat/ori/'+file)
    data.gen_mor()
    fname = file.split('.')[0] + '.pic'
    with gzip.open('ar/dat/morp/' + fname, 'wb') as f:
      pickle.dump(data, f)


def temp2(file):
  import os, pickle, gzip
  print(file)
  data = fl.read_csv('ar/dat/ori/'+file)
  data.gen_mor()
  fname = file.split('.')[0] + '_2' + '.pic'
  with gzip.open('ar/dat/morp/' + fname, 'wb') as f:
    pickle.dump(data, f)

def generate_original_img(file):
  print(file)
  data = fl.read_morp('ar/dat/morp/'+file)
  fname = file.split('.')[0]
  od = onset_dividing(data, 'abp')
  id = interpolation(od)
  nd = normalization(id)
  imgs = sgn_to_img(nd)
  imgs_abp = np.array(imgs)
  #np.savez_compressed('ar/dat/img(npz)/ori/'+fname+'_abp', imgs=imgs)

  od = onset_dividing(data, 'icp')
  id = interpolation(od)
  nd = normalization(id)
  imgs = sgn_to_img(nd)
  imgs_icp = np.array(imgs)
  np.savez_compressed('ar/dat/img(npz)/ori/'+fname, abp=imgs_abp, icp=imgs_icp)


def temp():
  import os
  files = os.listdir('ar/dat/real_ori/')
  for file in files:
    f = open('ar/dat/real_ori/' + file)
    lines = f.readlines()
    f.close()
    pen = open('ar/dat/cnv/' + file, 'w')
    for line in lines:
      sl = line.split(',')
      pen.write(sl[0] + ',' + sl[1] + ',' + sl[2] + '\n')
    pen.close()

def pulse_count():
  path = 'ar/dat/ori/'
  import os
  files = os.listdir(path)
  total_abp_pulse = 0
  total_icp_pulse = 0
  for file in files:
    print(file)
    data = fl.read_csv(path + file)
    data.gen_mor()
    total_abp_pulse += len(data.abp_ons)
    total_icp_pulse += len(data.icp_ons)

  print('abp\t' + str(total_abp_pulse))
  print('icp\t' + str(total_icp_pulse))
  


if __name__ == '__main__':
  pulse_count()
  #generate_original_img('299_1.pic')
  #temp2('40_1.csv')
  #save_morphological_feature()
  #representative_train()
  #data = fl.read_csv('E:/Richard/TBI/testdata/215_1.csv')
  #data.gen_mor()
  #data.show_abp_morp()
  #prd_abp = abp_proc(data)
  #prd_icp = icp_proc(data)
  #data.save('lll', prd_abp, prd_icp)

  #print('abc')

