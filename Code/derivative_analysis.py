import numpy as np
import peak_detection as pd
import os
import file_loader as fl
import matplotlib.pyplot as plt

class DAP:
  def __init__(self, signal):
    self.signal = signal
    self.first

def derivative(x, y):
  dy = np.zeros(y.shape, np.float)
  dy[:-1] = np.diff(y)/np.diff(x)
  dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
  return dy

def derivative(y):
  x = np.linspace(0, len(y), len(y))
  dy = np.zeros(y.shape, np.float)
  dy[:-1] = np.diff(y)/np.diff(x)
  dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
  return dy

def gen_abp_figure():
  data = fl.read_csv('E:/Richard/TBI/testdata/5_1.csv')
  data.abp = data.abp[:500]
  data.icp = data.icp[:500]
  fs = data.get_sampling_frequency()
  sys = pd.abp_sys_peak(data)
  arr = np.array(data.abp)
  x = np.linspace(0, len(arr) - 1, len(arr))
  sysx, sysy = pd.xy_list(sys, arr)
  ons = pd.get_pul_onset(arr, sys)
  onsx, onsy = pd.xy_list(ons, arr)
  nth = pd.get_dic_notch(arr, ons)
  nthx, nthy = pd.xy_list(nth, arr)
  plt.subplot(3, 1, 1)
  plt.plot(x, arr, 'b-', sysx, sysy, 'r*', onsx, onsy, 'g*', nthx, nthy, 'm*')

  dy = derivative(arr)
  ddy = derivative(dy)

  plt.subplot(3, 1, 2)
  dsys = pd.get_sys_peak(dy, fs)
  dsysx, dsysy = pd.xy_list(dsys, dy)
  dft = pd.get_first_through(dy, dsys)
  dftx, dfty = pd.xy_list(dft, dy)
  plt.plot(np.linspace(0, len(dy) - 1, len(dy)), dy, 'b-', dsysx, dsysy, 'r*', dftx, dfty, 'g*')

  plt.subplot(3, 1, 3)
  ddsys = pd.get_sys_peak(ddy, fs)
  ddsysx, ddsysy = pd.xy_list(ddsys, ddy)
  ddft = pd.get_first_through(ddy, ddsys)
  ddftx, ddfty = pd.xy_list(ddft, ddy)
  plt.plot(np.linspace(0, len(ddy) - 1, len(ddy)), ddy, 'b-', ddsysx, ddsysy, 'r*', ddftx, ddfty, 'g*')

  plt.show()

def smoothing2(data):
  win_len = 10
  s = np.r_[data[10-1:0:-1], data, data[-1:-10-1:-1]]
  w = eval('np.hanning(win_len)')
  y = np.convolve(w/w.sum(), s, mode='valid')
  return y

def gen_icp_figure():
  data = fl.read_csv('E:/Richard/TBI/testdata/5_1.csv')
  data.abp = data.abp[1500:2000]
  data.icp = data.icp[1500:2000]
  fs = data.get_sampling_frequency()
  data.gen_mor()
  ons = data.icp_ons[:-1]
  arr = np.array(smoothing2(data.icp))
  x = np.linspace(0, len(arr) - 1, len(arr))
  onsx, onsy = pd.xy_list(ons, arr)

  plt.subplot(3, 1, 1)
  plt.plot(x, arr, 'b-', onsx, onsy, 'r*')

  dy = derivative(arr)
  ddy = derivative(dy)

  plt.subplot(3, 1, 2)
  dsys = pd.get_sys_peak(dy, fs)
  dons = pd.get_pul_onset(dy, dsys)
  dnth = pd.get_dic_notch(dy, dons)
  dsysx, dsysy = pd.xy_list(dsys, dy)
  donsx, donsy = pd.xy_list(dons, dy)
  dnthx, dnthy = pd.xy_list(dnth, dy)
  #dft = pd.get_first_through(dy, dsys)
  #dftx, dfty = pd.xy_list(dft, dy)
  #plt.plot(np.linspace(0, len(dy) - 1, len(dy)), dy, 'b-', dsysx, dsysy, 'r*', dftx, dfty, 'g*')
  plt.plot(np.linspace(0, len(dy) - 1, len(dy)), dy, 'b-', dsysx, dsysy, 'r*', donsx, donsy, 'g*', dnthx, dnthy, 'c*')
  plt.subplot(3, 1, 3)
  nth_ori_x, nth_ori_y = pd.xy_list(dnth, arr)
  sys_ori_x, sys_ori_y = pd.xy_list(dsys, arr)
  ons_ori_x, ons_ori_y = pd.xy_list(dons, arr)
  plt.plot(x, arr, 'b-', nth_ori_x, nth_ori_y, 'c*', sys_ori_x, sys_ori_y, 'r*', ons_ori_x, ons_ori_y, 'g*')
  
  #ddsys = pd.get_sys_peak(ddy, fs)
  #ddsysx, ddsysy = pd.xy_list(ddsys, ddy)
  #ddft = pd.get_first_through(ddy, ddsys)
  #ddftx, ddfty = pd.xy_list(ddft, ddy)
  #plt.plot(np.linspace(0, len(ddy) - 1, len(ddy)), ddy, 'b-', ddsysx, ddsysy, 'r*', ddftx, ddfty, 'g*')

  plt.show()




if __name__ == '__main__':
  gen_icp_figure()