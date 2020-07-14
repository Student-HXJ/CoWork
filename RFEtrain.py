import time
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from datautils import data1set, labels, plotroc


class RFEtrain:

    def __init__(self, dataset, label):

        # model
        self.model = SVC(C=1, kernel="linear")
        # dataset
        self.dataset = dataset
        self.labels = label

    def RFEprocess(self, features):
        selector = RFE(self.model, n_features_to_select=features)
        selector = selector.fit(self.dataset, self.labels)
        summary = np.zeros(sum(selector.support_)).tolist()
        j = 0
        k = 0
        for i in selector.support_:
            j = j + 1
            if i:
                summary[k] = j - 1
                k = k + 1
        data = self.dataset[:, summary]
        return summary, data

    def feature_select(self):
        for i in range(1, 101, 1):
            start = time.time()
            idx, RFEdata = self.RFEprocess(i)
            acc = cross_val_score(self.model, RFEdata, self.labels, cv=10, scoring='accuracy').mean()
            end = time.time()
            print("time: {:.2f}s, featurenum: {:d}, acc: {:.4f}".format(end - start, i, acc))

    def cross_validation(self):
        kf = KFold(n_splits=49)
        idx, RFEdata = self.RFEprocess(10)

        labelset = []
        scoreset = []
        predset = []
        for train, test in kf.split(RFEdata):
            trainset = RFEdata[train]
            trainlabel = labels[train]
            testset = RFEdata[test]
            testlabel = labels[test]

            self.model.fit(trainset, trainlabel)
            score = self.model.decision_function(testset)
            pred = self.model.predict(testset)

            labelset.append(testlabel[0])
            scoreset.append(score[0])
            predset.append(pred[0])

        acc = accuracy_score(labelset, predset)
        print(acc)
        plotroc(labelset, scoreset)
        print("done!")


if __name__ == "__main__":
    RFEtrain(data1set, labels).feature_select()
