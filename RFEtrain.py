from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import time
from sklearn.svm import SVC
from datautils import getacc, negtozero
import os


class RFEtrain:
    def __init__(self, dataset, labels):

        # model
        self.model = SVC(C=1, kernel="linear")
        # dataset
        self.dataset = dataset
        self.labels = labels
        self.samples = dataset.shape[0]
        self.features = dataset.shape[1]

    def RFEprocess(self, data, label, train_idx, test_idx, features):
        selector = RFE(self.model, n_features_to_select=features)
        selector = selector.fit(data[train_idx], label[train_idx])

        summary = np.zeros(sum(selector.support_)).tolist()
        j = 0
        k = 0
        for i in selector.support_:
            j = j + 1
            if i:
                summary[k] = j - 1
                k = k + 1

        self.model.fit(data[train_idx][:, summary], label[train_idx])

        pred = self.model.predict(data[test_idx][:, summary])
        score = self.model.decision_function(data[test_idx][:, summary])
        acc = accuracy_score(label[test_idx], pred)

        result = np.array([acc, score, label[test_idx]])
        return result

    def cross_validation(self, features):
        result = Parallel(16)(delayed(self.RFEprocess)(
            self.dataset,
            self.labels,
            train_idx,
            test_idx,
            features,
        ) for train_idx, test_idx in KFold(n_splits=self.samples).split(self.labels))

        acc = getacc(result)
        # getroc(result)
        return acc

    def feature_select(self):
        for i in range(100, self.features + 1, 5):
            start = time.time()
            acc = self.cross_validation(i)
            end = time.time()
            print("time: {:.2f}s, featurenum: {:d}, acc: {:.2f}".format(end - start, i, acc))


class RFECVtrain:
    def __init__(self, dataset, labels):
        self.model = SVC(C=1, kernel='linear')
        self.dataset = dataset
        print(dataset.shape)
        self.label = labels

    def useRFECV(self, savepath):

        rfecv = RFECV(estimator=self.model, step=1, cv=KFold(75), scoring='accuracy')
        rfecv.fit(self.dataset, self.label)

        plt.xlabel("number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.savefig(savepath)

        bestcnt = 0
        bestscores = 0
        for i in range(len(rfecv.grid_scores_)):
            if rfecv.grid_scores_[i] > bestscores:
                bestscores = rfecv.grid_scores_[i]
                bestcnt = i + 1

        print(bestcnt, bestscores)


if __name__ == "__main__":
    dirpath = os.getcwd()
    data1path = os.path.join(dirpath, 'dataset', '75', 'data1.npy')
    data2path = os.path.join(dirpath, 'dataset', '75', 'data2.npy')
    labelpath = os.path.join(dirpath, 'dataset', '75', 'label.npy')
    data1 = np.load(data1path)
    data2 = np.load(data2path)
    label = negtozero(np.load(labelpath))

    RFECVtrain(data1, label).useRFECV('./image/RFECVtrain_75_test1.jpg')
    RFECVtrain(data2, label).useRFECV('./image/RFECVtrain_75_test2.jpg')
