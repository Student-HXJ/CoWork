import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datautils import negtozero

dirpath = os.getcwd()
datapath = os.path.join(dirpath, 'dataset', '75f', 'data2.npy')
labelpath = os.path.join(dirpath, 'dataset', '75f', 'label.npy')

data = np.load(datapath)
label = negtozero(np.load(labelpath))
print(data.shape)

model = SVC(kernel='rbf', C=10)

acc = 0
cnt = 0
for train_idx, test_idx in KFold(75).split(label):
    model.fit(data[train_idx], label[train_idx])
    pred = model.predict(data[test_idx])
    acc += accuracy_score(label[test_idx], pred)
    cnt += 1

print(acc / cnt)
