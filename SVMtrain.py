import os
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from datautils import plot_roc_auc
from matplotlib import pyplot as plt


class SVMtrain:
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
        self.savepath = os.path.join(dirpath, 'image', kinds + '_svc.jpg')

        data1 = np.load(data1path)
        data2 = np.load(data2path)
        self.label = np.load(labelpath)

        self.trainproc(data1)
        self.trainproc(data2)

        plt.cla()

    def trainproc(self, data):

        acc = 0
        cnt = 0
        y_label = []
        y_score = []
        model = self._grid_search(data)
        for train_idx, test_idx in KFold(len(data)).split(self.label):

            random.shuffle(train_idx)
            model.fit(data[train_idx], self.label[train_idx])
            pred = model.predict(data[test_idx])
            score = model.decision_function(data[test_idx])
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

    SVMtrain('75f')
    SVMtrain('75')
    SVMtrain('49')
