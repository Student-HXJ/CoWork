from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from datautils import plot_roc_auc
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
import os
import random


class RFEtrain:
    def __init__(self, kinds):

        self.svm_param = [{
            "kernel": ['linear'],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
        }, {
            "kernel": ['poly'],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
            'gamma': ['scale', 'auto'],
            'degree': [i for i in range(1, 11)],
        }, {
            "kernel": ['rbf'],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
            'gamma': ['scale', 'auto'],
        }, {
            "kernel": ['sigmoid'],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
            'gamma': ['scale', 'auto'],
        }]

        self.svm = SVC()

        dirpath = os.getcwd()
        data1path = os.path.join(dirpath, 'dataset', kinds, 'data1.npy')
        data2path = os.path.join(dirpath, 'dataset', kinds, 'data2.npy')
        labelpath = os.path.join(dirpath, 'dataset', kinds, 'label.npy')
        self.savepath = os.path.join(dirpath, 'image', kinds + '_rfe.jpg')

        data1 = np.load(data1path)
        data2 = np.load(data2path)
        self.label = np.load(labelpath)

        print(data1.shape)
        self.useRFECV(data1)
        print(data2.shape)
        self.useRFECV(data2)

        plt.cla()

    def useRFECV(self, data):

        model = self._grid_search(data)
        model_ = model

        model = RFECV(estimator=model, step=1, cv=KFold(len(data)), scoring='accuracy', n_jobs=-1)

        model.fit(data, self.label)
        plt.xlabel("number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(model.grid_scores_) + 1), model.grid_scores_)
        plt.savefig(self.savepath)
        bestcnt = 0
        bestscores = 0
        for i in range(len(model.grid_scores_)):
            if model.grid_scores_[i] > bestscores:
                bestscores = model.grid_scores_[i]
                bestcnt = i + 1
                support_ = model.support_

        print(bestcnt, bestscores)

        self.RFEproc(data, model_, self.get_summary(support_))

    def get_summary(self, support):

        summary = np.zeros(sum(support)).tolist()
        j = 0
        k = 0
        for i in support:
            j = j + 1
            if i:
                summary[k] = j - 1
                k = k + 1
        return summary

    def RFEproc(self, data, model, summary):

        acc = 0
        cnt = 0
        y_label = []
        y_score = []
        for train_idx, test_idx in KFold(len(data)).split(self.label):
            random.shuffle(train_idx)
            model.fit(data[train_idx][:, summary], self.label[train_idx])
            pred = model.predict(data[test_idx][:, summary])
            score = model.decision_function(data[test_idx][:, summary])
            y_score.append(score)
            y_label.append(self.label[test_idx])
            acc += accuracy_score(self.label[test_idx], pred)
            cnt += 1
        print(data.shape, acc / cnt)
        plot_roc_auc(y_label, y_score, self.savepath)

    def _grid_search(self, data):
        grid_search = GridSearchCV(self.svm, self.svm_param, n_jobs=-1, cv=KFold(len(data)))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":
    # RFEtrain('75f')
    RFEtrain('75')
    # RFEtrain('49')
