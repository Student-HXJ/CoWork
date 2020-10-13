# import torch
# a = torch.ones(2,2,5)
# print(a)
# print(a[1,:,:])

import os
filepath = './China_PM25_T'
files = []
for e in os.listdir(filepath):
  files.append(e)
print(files)
print()
