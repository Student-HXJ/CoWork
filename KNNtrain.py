import os
import numpy as np
import random
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datautils import plot_roc_auc
from matplotlib import pyplot as plt


class KNNtrain():
    def __init__(self, kinds):
        # Grid Search
        self.knn_param = {
            "weights": ['uniform', 'distance'],
            'n_neighbors': [i for i in range(1, 11)],
        }

        dirpath = os.getcwd()
        data1path = os.path.join(dirpath, 'dataset', kinds, 'data1.npy')
        data2path = os.path.join(dirpath, 'dataset', kinds, 'data2.npy')
        labelpath = os.path.join(dirpath, 'dataset', kinds, 'label.npy')
        self.savepath = os.path.join(dirpath, 'image', kinds + '_knn.jpg')

        data1 = np.load(data1path)
        data2 = np.load(data2path)
        self.label = np.load(labelpath)

        self.knn = KNeighborsClassifier()

        self.trainproc(data1)
        self.trainproc(data2)

        plt.cla()

    def trainproc(self, data):

        acc = 0
        cnt = 0
        y_label = []
        y_score = []
        model = self._grid_search(self.knn, data)
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(self.label):

            random.shuffle(train_idx)
            model.fit(data[train_idx], self.label[train_idx])
            pred = model.predict(data[test_idx])
            score = model.predict_proba(data[test_idx])[:, 1]
            y_score.append(score)
            y_label.append(self.label[test_idx])
            acc += accuracy_score(self.label[test_idx], pred)
            cnt += 1

        print(data.shape, acc / cnt)
        plot_roc_auc(y_label, y_score, self.savepath)

    def _grid_search(self, model, data):
        grid_search = GridSearchCV(model, self.knn_param, n_jobs=-1, cv=KFold(len(data), shuffle=True))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":

    KNNtrain('75f')
    KNNtrain('75')
    KNNtrain('49')
