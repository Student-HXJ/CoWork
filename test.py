from datautils import get_data
import os

dirpath = os.getcwd()
savepath = os.path.join(dirpath, 'dataset', '75n', 'data2.npy')

data1, data2, label = get_data('75n')
print(data1.shape)
print(data2.shape)
print(label.shape)
