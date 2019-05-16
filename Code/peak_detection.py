
import os
import numpy as np
import matplotlib.pyplot as plt
import signals as sn
import file_loader as fl


def rev_near_peak(idx, data, fs):
  near_peak = []
  for i in range(len(idx) - 1):
    if idx[i+1] - idx[i] < (fs * 0.1):
      if data[idx[i]] > data[idx[i + 1]]:
        near_peak.append(idx[i+1])
      else:
        near_peak.append(idx[i])
  near_peak = list(set(near_peak))
  for v in near_peak:
    idx.remove(v)
  return np.array(idx)
#region Basic morphology
def get_sys_peak(data, fs):
  arr = np.array(data)
  i = 0
  sys_idx = []
  while(i + int(fs) < len(arr)):
    max_idx = arr[i:i+int(fs)].argmax()
    sys_idx.append(max_idx + i)
    i += int(fs/10)
  sys_idx = list(set(sys_idx))
  sys_idx.sort()
  sys_idx = rev_near_peak(sys_idx, arr, fs)
  sys_idx = np.array(list(set(sys_idx)))
  sys_idx.sort()
  return sys_idx

def get_pul_onset(data, sys):
  ons = []
  fs = sys[2] - sys[1]
  for i in range(len(sys)):
    j = int(fs/20)
    while sys[i] - j > 0:
      if data[sys[i] - j] >= data[sys[i] - j + 1] and data[sys[i] - j + 1] < data[sys[i] - j + 2]:
        if data[sys[i]] - data[sys[i] - j + 1] >= (data[sys[i]]/20):
          break
      j += 1
    if sys[i] - j <= 0 or j > int(fs/2):
      continue
    else:
      ons.append(sys[i] - j + 1)
  ons = np.array(list(set(ons)))
  ons.sort()
  return ons 

def get_dic_notch(data, ons):
  arr = np.array(data)
  nth = []
  for i in range(len(ons) - 1):
    str_idx = int((ons[i+1] - ons[i]) / 3)
    j = ons[i] + str_idx
    while j < ons[i+1]:
      if arr[j] < arr[j + 1] and arr[j] < arr[j - 1]:
        break
      j += 1
    if j >= ons[i+1]:
      j = ons[i] + str_idx
      max_j = 0; max_angle = 0;
      while j < ons[i+1]:
        if (arr[j-1] - arr[j]) - (arr[j] - arr[j+1]) > max_angle:
          max_angle = (arr[j-1] - arr[j]) - (arr[j] - arr[j+1])
          max_j = j
        j += 1
      nth.append(max_j)
    else:
      nth.append(j)
  nth = np.array(list(set(nth)))
  nth.sort()
  return nth
#endregion

def get_gradient(x, y, i):
  if i < 2 or i > len(x) - 2: return 0
  h = (x[i - 1] - x[i - 2]).total_seconds() * 100
  return (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12 * h)

def get_icp_onset(data):
  fwd_ang_dic = {}
  bwd_ang_dic = {}
  abwd_ang_dic = {}
  ons_list = []
  fs = data.get_sampling_frequency()
  ons_len_sum = 0; ons_len_cnt = 0;
  for i in range(int(fs * 0.4), len(data.icp) - int(fs * 0.4)):
    fst_idx = i - int(fs * 0.4)
    lst_idx = i + int(fs * 0.4)
    fwd_ang_sum = 0
    for j in range(i + 1, lst_idx):
      x_diff = (data.dtime[j]-data.dtime[i]).total_seconds()*10
      y_diff = data.icp[j] - data.icp[i]
      cur_ang = np.arctan(y_diff / x_diff) * (180 / np.pi)
      fwd_ang_sum += cur_ang
    abwd_ang_sum = 0; bwd_ang_sum = 0;
    for j in range(fst_idx, i):
      x_diff = (data.dtime[j]-data.dtime[i]).total_seconds()*10
      y_diff = data.icp[j] - data.icp[i]
      cur_ang = np.arctan(y_diff / x_diff) * (180 / np.pi)
      abwd_ang_sum += np.abs(cur_ang)
      bwd_ang_sum += cur_ang
    fwd_ang_dic[i] = fwd_ang_sum
    bwd_ang_dic[i] = bwd_ang_sum
    abwd_ang_dic[i] = abwd_ang_sum
  for i in range(1, len(data.abp_ons)):
    fst_idx = data.abp_ons[i - 1]
    lst_idx = data.abp_ons[i]
    candidate_onset = []
    for j in range(fst_idx, lst_idx):
      pre_grad = get_gradient(data.dtime, data.icp, j - 1)
      now_grad = get_gradient(data.dtime, data.icp, j)
      if now_grad == 0: candidate_onset.append(j)
      elif pre_grad < 0 and now_grad > 0:
        if data.icp[j] <= data.icp[j - 1]:
          candidate_onset.append(j)
        else: candidate_onset.append(j-1)
    for_p1_val = -10000; for_p1_idx = 0;
    for_p2_val = -10000; for_p2_idx = 0;

    for j in candidate_onset:
      if j not in fwd_ang_dic or j not in bwd_ang_dic: continue
      if fwd_ang_dic[j] + abwd_ang_dic[j] > for_p2_val:
        for_p2_val = fwd_ang_dic[j] + abwd_ang_dic[j]
        for_p2_idx = j
      if fwd_ang_dic[j] - abwd_ang_dic[j] > for_p1_val:
        for_p1_val = fwd_ang_dic[j] - abwd_ang_dic[j]
        for_p1_idx = j
    if len(candidate_onset) < 1:
      ons_list.append(2*ons_list[-1] - ons_list[-2])
    elif len(ons_list) < 1:
      if data.icp[for_p1_idx] < data.icp[for_p2_idx]:
        ons_list.append(for_p1_idx)
      else:
        ons_list.append(for_p2_idx)
    elif len(ons_list) < 5:
      p1_len = (data.dtime[for_p1_idx] - data.dtime[ons_list[-1]]).total_seconds()
      p2_len = (data.dtime[for_p2_idx] - data.dtime[ons_list[-1]]).total_seconds()
      is_p1_abnormal = False; is_p2_abnormal = False;
      if p1_len < 0.4 or p1_len > 1.2:
        is_p1_abnormal = True
      if p2_len < 0.4 or p2_len > 1.2:
        is_p1_abnormal = True
      if is_p1_abnormal and not is_p2_abnormal:
        ons_list.append(for_p2_idx)
        ons_len_sum += (data.dtime[for_p2_idx] - data.dtime[ons_list[-1]]).total_seconds()
        ons_len_cnt += 1
      elif not is_p1_abnormal and is_p2_abnormal:
        ons_list.append(for_p1_idx)
        ons_len_sum += (data.dtime[for_p1_idx] - data.dtime[ons_list[-1]]).total_seconds()
        ons_len_cnt += 1
      else:
        if data.icp[for_p1_idx] < data.icp[for_p1_idx]:
          ons_list.append(for_p1_idx)
          ons_len_sum += (data.dtime[for_p1_idx] - data.dtime[ons_list[-1]]).total_seconds()
          ons_len_cnt += 1
        else:
          ons_list.append(for_p2_idx)
          ons_len_sum += (data.dtime[for_p2_idx] - data.dtime[ons_list[-1]]).total_seconds()
          ons_len_cnt += 1
    else:
      p1_len = (data.dtime[for_p1_idx] - data.dtime[ons_list[-1]]).total_seconds()
      p2_len = (data.dtime[for_p2_idx] - data.dtime[ons_list[-1]]).total_seconds()
      oal = ons_len_sum / ons_len_cnt
      if np.abs(p1_len - oal) > np.abs(p2_len - oal):
        ons_list.append(for_p2_idx)
        ons_len_sum += (data.dtime[for_p2_idx] - data.dtime[ons_list[-1]]).total_seconds()
        ons_len_cnt += 1
      else:
        ons_list.append(for_p1_idx)
        ons_len_sum += (data.dtime[for_p1_idx] - data.dtime[ons_list[-1]]).total_seconds()
        ons_len_cnt += 1

  ons_list = np.array(list(set(ons_list)))
  ons_list.sort()
  ons_list = rev_near_peak(list(ons_list), data.icp, data.get_sampling_frequency())
  return ons_list

def get_first_through(data, sys):
  arr = np.array(data)
  fs = sys[2] - sys[1]
  ft = []
  for i in range(len(sys)):
    j = 0
    while sys[i] + j < len(arr):
      if data[sys[i] + j] >= data[sys[i] + j + 1] and data[sys[i] + j + 1] < data[sys[i] + j + 2]:
        if data[sys[i]] - data[sys[i] + j + 1] >= (data[sys[i]]/20):
          break
      j += 1
    if sys[i] + j <= 0 or j > int(fs/2):
      continue
    else:
      ft.append(sys[i] + j + 1)
  ft.sort()
  return np.array(ft)


def get_first_peak(data, ft):
  from derivative_analysis import derivative
  dy = derivative(data)












def abp_sys_peak(data):
  arr = np.array(data.abp)
  fs = data.get_sampling_frequency()
  return get_sys_peak(arr, fs)

def xy_list(idx, data):
  xs = []; ys =[];
  for i in idx:
    xs.append(i)
    ys.append(data[i])
  return xs, ys

if __name__ == '__main__':
  d = fl.read_csv('E:/Richard/TBI/testdata/5_1.csv')
  v = abp_sys_peak(d)
  x, y = xy_list(v, d.abp)
  plt.plot(x[:10], y[:10], 'r*', np.linspace(0, 320, 321), d.abp[:321])
  #plt.plot([2, 5, 6], [2, 5, 3], 'r*')
  plt.show()