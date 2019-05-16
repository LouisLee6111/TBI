import numpy as np
import signals as sig
from scipy.stats.stats import pearsonr


def moving_average(arr, win_sam, mov_sam):
  i = 0
  avg_arr = []
  while i * mov_sam + win_sam < len(arr):
    avg_arr.append(np.average(arr[i * mov_sam:i * mov_sam + win_sam]))
    i += 1
  return avg_arr

def moving_average_sig(x_arr, y_arr, win_sam, mov_sam):
  i = 0
  avg_ssig = sig.Sim_Signal()
  while i * mov_sam + win_sam < len(x_arr):
    cur_x = np.average(x_arr[i * mov_sam:i * mov_sam + win_sam])
    cur_y = np.average(y_arr[i * mov_sam:i * mov_sam + win_sam])
    avg_ssig.append(cur_x, cur_y)
    i += 1
  return avg_ssig

def moving_correlation(arr1, arr2, win_sam, mov_sam):
  assert len(arr1) == len(arr2), 'Length of two input arrays are different!'
  i = 0
  cor_arr = []
  while i * mov_sam + win_sam < len(arr1):
    corr = pearsonr(arr1[i * mov_sam:i * mov_sam + win_sam], arr2[i * mov_sam:i * mov_sam + win_sam])
    cor_arr.append(corr[0])
    i += 1
  return cor_arr

def moving_substraction(arr1, arr2, win_sam, mov_sam):
  assert len(arr1) == len(arr2), 'Length of two input arrays are different!'
  i = 0
  sub_arr = []
  while i * mov_sam + win_sam < len(arr1):
    sub = np.average(arr1[i * mov_sam:i * mov_sam + win_sam]) - np.average(arr2[i * mov_sam:i * mov_sam + win_sam])
    sub_arr.append(sub)
    i += 1
  return sub_arr