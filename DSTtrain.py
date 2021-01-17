import os
from datautils import get_data, plot_roc_auc
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold


class DSTtrain():
    def __init__(self, kinds):

        self.savepath = os.path.join(os.getcwd(), 'image', kinds + '_dst.jpg')

        data1, data2, self.label = get_data(kinds)

        self.dst = DecisionTreeClassifier(random_state=1)

        self.trainproc(data1)
        self.trainproc(data2)

        plt.close()

    def trainproc(self, data):

        dst_param = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None] + [i for i in range(1, 10)],
            'max_features': [i for i in range(1, data.shape[1])],
        }

        acc = 0
        y_label = []
        y_score = []

        model = self._grid_search(self.dst, data, dst_param)
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(data):

            model.fit(data[train_idx], self.label[train_idx])
            acc += model.score(data[test_idx], self.label[test_idx])

            y_label.append(self.label[test_idx])
            y_score.append(model.predict_proba(data[test_idx])[:, 1])

        print(data.shape, acc / len(data))
        plot_roc_auc(y_label, y_score, self.savepath)

    def _grid_search(self, model, data, param):
        grid_search = GridSearchCV(model, param, n_jobs=-1, cv=KFold(len(data)))
        grid_search.fit(data, self.label)
        return grid_search.best_estimator_


if __name__ == "__main__":

    DSTtrain('75n')
    DSTtrain('75f')
    DSTtrain('75')
    DSTtrain('49')
