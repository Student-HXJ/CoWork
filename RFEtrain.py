from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from datautils import plot_roc_auc
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
import os


class RFEtrain:
    def __init__(self, kinds):

        self.svm_param = {
            'kernel': ['linear'],
            'gamma': ['scale', 'auto'],
            'shrinking': [False, True],
            'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
        }

        self.svm = SVC()

        dirpath = os.getcwd()
        data1path = os.path.join(dirpath, 'dataset', kinds, 'data1.npy')
        data2path = os.path.join(dirpath, 'dataset', kinds, 'data2.npy')
        labelpath = os.path.join(dirpath, 'dataset', kinds, 'label.npy')
        self.cvSavepath = os.path.join(dirpath, 'image', kinds + '_rfecv.jpg')
        self.fsSavepath = os.path.join(dirpath, 'image', kinds + '_rfefs.jpg')

        data1 = np.load(data1path)
        data2 = np.load(data2path)
        self.label = np.load(labelpath)

        self.RFEtrain(data1)
        self.RFEtrain(data2)

        plt.cla()

    def RFEtrain(self, data):

        search_model = self._grid_search(data)
        model_ = search_model
        model = RFECV(estimator=search_model, step=1, cv=KFold(len(data), shuffle=True), scoring='accuracy', n_jobs=-1)
        X = model.fit_transform(data, self.label)

        plt.xlabel("number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(model.grid_scores_) + 1), model.grid_scores_)
        plt.savefig(self.fsSavepath)

        self.trainproc(X, model_)

    def trainproc(self, data, model):

        acc = 0
        cnt = 0
        y_label = []
        y_score = []
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(data):
            model.fit(data[train_idx], self.label[train_idx])
            pred = model.predict(data[test_idx])
            score = model.decision_function(data[test_idx])
            y_score.append(score)
            y_label.append(self.label[test_idx])
            acc += accuracy_score(self.label[test_idx], pred)
            cnt += 1
        print(data.shape, acc / cnt)
        plot_roc_auc(y_label, y_score, self.cvSavepath)

    def _grid_search(self, data):
        grid_search = GridSearchCV(self.svm, self.svm_param, n_jobs=-1, cv=KFold(len(data), shuffle=True))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":
    RFEtrain('75f')
    RFEtrain('75')
    RFEtrain('49')
