import numpy as np

def load(file):
  f = open(file)
  line = f.readline()
  f.close()
  rd = line.split(',')
  s = []
  for r in rd:
    s.append(float(r))
  return s

if __name__ == '__main__':
  path = 'E:/Richard/TBI/simulator code/Multiscale-Intracranial-Pressure-Simulator-master/Multiscale-Intracranial-Pressure-Simulator-master/'
  artifact = load(path + 'artifact.csv')
  normal = load(path + 'normal.csv')
  import matplotlib.pyplot as plt
  
  plt.subplot(2, 1, 1)
  plt.plot(np.linspace(0, len(normal) - 1, len(normal)), normal, 'b-')

  plt.subplot(2, 1, 2)
  plt.plot(np.linspace(0, len(artifact) - 1, len(artifact)), artifact, 'b-')
  plt.show()
  