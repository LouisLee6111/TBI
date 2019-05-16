import pytz
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import peak_detection as pd
import artifact_removal as ar


def fromOADate(v):
  return datetime(1899,12,30,0,0,0,tzinfo=pytz.utc)+timedelta(days=v)

def toOADate(v):
  return (v - datetime(1899,12,30,0,0,0,tzinfo=pytz.utc)).total_secondse()/24/60/60

class Sim_Signal:
  def __init__(self):
    self.x = np.array([]); self.y = np.array([]);

  def __init__(self, x, y):
    self.x = x; slef.y = y;

  def append(self, x, y):
    self.x.append(x)
    self.y.append(y);

class Signal:
  def __init__(self, filename):
    self.filename = filename
    self.dtime = []; self.abp = []; self.icp = [];
    self.ldtime = []; self.abp_sys = []; self.abp_ons = [];
    self.icp_ons = []; self.abp_artifact = [];
    self.icp_artifact = [];

  def insert(self, line, a_idx, i_idx):
    sl = line.split(',')
    self.dtime.append(fromOADate(float(sl[0])))
    self.ldtime.append(float(sl[0]))
    try:
      self.abp.append(float(sl[a_idx]))
    except:
      self.abp.append(self.abp[-1])
    try:
      self.icp.append(float(sl[i_idx]))
    except:
      self.icp.append(self.icp[-1])
    
  def get_sampling_frequency(self):
    # Monte Carlo
    import random
    msf = 0
    for i in range(10):
      v = random.randint(0, len(self.dtime) - 1)
      intv = (self.dtime[v+1] - self.dtime[v]).total_seconds()
      msf += (1/intv)
    return msf/10

  def gen_mor(self):
    fs = self.get_sampling_frequency()
    self.abp_sys = pd.get_sys_peak(self.abp, fs)
    self.abp_ons = pd.get_pul_onset(self.abp, self.abp_sys)
    self.icp_ons = pd.get_icp_onset(self)

  def show_abp_morp(self):
    x0 = np.linspace(0, len(self.abp) - 1, len(self.abp))
    x1, y1 = pd.xy_list(self.abp_sys, self.abp)
    x2, y2 = pd.xy_list(self.abp_ons, self.abp)
    plt.plot(x0, self.abp, 'b-', x1, y1, 'r*', x2, y2, 'c*')
    plt.show()

  def save(self, path, abp_art, icp_art):
    #pen = open(path + self.filename)
    abp_idx = []; icp_idx = []
    for i in range(len(abp_art)):
      if abp_art[i] == 0: continue
      for idx in range(self.abp_ons[i],self.abp_ons[i+1]):
        abp_idx.append(idx)
    for i in range(len(icp_art)):
      if icp_art[i] == 0: continue
      for idx in range(self.icp_ons[i],self.icp_ons[i+1]):
        icp_idx.append(idx)
    idc = list(set(abp_idx) & set(icp_idx))

  def load_artifact_info(self, path, file, mode='abp'):
    ar_info_reader = open(path + file + '_1.ar.csv')
    lines = ar_info_reader.readlines()
    ar_info_reader.close()
    for line in lines[1:]:
      sl = line.split(',')
      if mode == 'abp':
        self.abp_artifact.append([float(sl[0]), float(sl[1])])
      elif mode =='icp':
        self.icp_artifact.append([float(sl[0]), float(sl[1])])
   
      


    
