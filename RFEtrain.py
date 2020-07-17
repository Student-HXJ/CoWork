import time
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from datautils import data1set, labels
from joblib import Parallel, delayed


class RFEtrain:

    def __init__(self, dataset, label):

        # model
        self.model = SVC(C=1, kernel="linear")
        # dataset
        self.dataset = dataset
        self.labels = label
        self.samples = dataset.shape[0]

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
        acc = accuracy_score(label[test_idx], pred)

        return acc

    def cross_validation(self, features):
        acc = Parallel(16)(delayed(self.RFEprocess)(
            self.dataset,
            self.labels,
            train_idx,
            test_idx,
            features,
        ) for train_idx, test_idx in KFold(n_splits=self.samples).split(self.dataset))
        acc = np.sum(acc) / self.samples
        return acc

    def feature_select(self):
        for i in range(1, self.dataset.shape[1] + 1, 1):
            start = time.time()
            acc = self.cross_validation(i)
            end = time.time()
            print("time: {:.2f}s, featurenum: {:d}, acc: {:.4f}".format(end - start, i, acc))


if __name__ == "__main__":
    RFEtrain(data1set, labels).feature_select()
