import signals as sn
import pytz
import numpy as np
from datetime import datetime, timedelta
import pickle, gzip


def fromOADate(v):
  return datetime(1899,12,30,0,0,0,tzinfo=pytz.utc)+timedelta(days=v)

def read_csv(file):
  f = open(file)
  lines = f.readlines()
  f.close()
  header = lines[0].split(',')
  for i in range(len(header)):
    if 'abp' in header[i]:
      abp_idx = i
    if 'icp' in header[i]:
      icp_idx = i
  tf_data = sn.Signal(file.split('/')[-1])
  for line in lines[1:]:
    tf_data.insert(line, abp_idx, icp_idx)
  return tf_data

def read_morp(file):
  with gzip.open(file, 'rb') as f:
    data = pickle.load(f)
  return data

def save_morp(data, file):
  with gzip.open(file, 'wb') as f:
    pickle.dump(data, f)
  return


if __name__ == '__main__':
  d = read_csv('E:/Richard/TBI/testdata/5_1.csv')
  ac = d.get_sampling_frequency()
  