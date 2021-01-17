import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from datautils import plot_roc_auc, get_data
from matplotlib import pyplot as plt


class KNNtrain():
    def __init__(self, kinds):
        # Grid Search
        self.knn_param = {
            "weights": ['uniform', 'distance'],
            'n_neighbors': [i for i in range(1, 11)],
        }

        self.savepath = os.path.join(os.getcwd(), 'image', kinds + '_knn.jpg')

        data1, data2, self.label = get_data(kinds)

        self.knn = KNeighborsClassifier()

        self.trainproc(data1)
        self.trainproc(data2)

        plt.close()

    def trainproc(self, data):

        acc = 0
        y_label = []
        y_score = []
        model = self._grid_search(self.knn, data)
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(data):

            model.fit(data[train_idx], self.label[train_idx])
            acc += model.score(data[test_idx], self.label[test_idx])
            y_score.append(model.predict_proba(data[test_idx])[:, 1])
            y_label.append(self.label[test_idx])

        print(data.shape, acc / len(data))
        plot_roc_auc(y_label, y_score, self.savepath)

    def _grid_search(self, model, data):
        grid_search = GridSearchCV(model, self.knn_param, n_jobs=-1, cv=KFold(len(data)))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":

    KNNtrain('75n')
    KNNtrain('75f')
    KNNtrain('75')
    KNNtrain('49')
