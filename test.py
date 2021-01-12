import os
import numpy as np
dirpath = os.getcwd()
datapath = os.path.join(dirpath, 'dataset', '75f', 'data2.npy')

data = np.load(datapath)
print(data.shape)
