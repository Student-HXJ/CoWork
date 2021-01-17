import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from datautils import plot_roc_auc, get_data
from matplotlib import pyplot as plt


class SVMtrain:
    def __init__(self, kinds):

        self.svm_param = [
            {
                "kernel": ['linear', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
            },
            {
                "kernel": ['poly'],
                'gamma': ['scale', 'auto'],
                'degree': [i for i in range(1, 11)],
                'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
            },
        ]

        self.svm = SVC()

        self.savepath = os.path.join(os.getcwd(), 'image', kinds + '_svc.jpg')

        data1, data2, self.label = get_data(kinds)

        self.trainproc(data1)
        self.trainproc(data2)

        plt.close()

    def trainproc(self, data):

        acc = 0
        y_label = []
        y_score = []
        model = self._grid_search(data)
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(data):

            model.fit(data[train_idx], self.label[train_idx])
            acc += model.score(data[test_idx], self.label[test_idx])
            y_score.append(model.decision_function(data[test_idx]))
            y_label.append(self.label[test_idx])

        print(data.shape, acc / len(data))
        plot_roc_auc(y_label, y_score, self.savepath)

    def _grid_search(self, data):
        grid_search = GridSearchCV(self.svm, self.svm_param, n_jobs=-1, cv=KFold(len(data)))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":

    SVMtrain('75n')
    SVMtrain('75f')
    SVMtrain('75')
    SVMtrain('49')
