import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datautils import negtozero


class SVMCVtrain:
    def __init__(self, dataset, label):

        self.model = SVC(kernel='rbf', C=10)
        self.data = dataset
        self.label = label

    def trainproc(self):

        acc = 0
        cnt = 0
        for train_idx, test_idx in KFold(75).split(self.label):
            self.model.fit(self.data[train_idx], self.label[train_idx])
            pred = self.model.predict(self.data[test_idx])
            acc += accuracy_score(self.label[test_idx], pred)
            cnt += 1

        print(acc / cnt)


if __name__ == "__main__":
    dirpath = os.getcwd()
    datapath = os.path.join(dirpath, 'dataset', '75f', 'data2.npy')
    labelpath = os.path.join(dirpath, 'dataset', '75f', 'label.npy')

    data = np.load(datapath)
    label = negtozero(np.load(labelpath))
    print(data.shape)

    SVMCVtrain(data, label).trainproc()
