from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, GridSearchCV
from datautils import plot_roc_auc, plot_feature_selected, get_data
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import os


class RFECVtrain:
    def __init__(self, kinds):

        self.svm_param = {
            'gamma': ['scale', 'auto'],
            'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
        }

        self.svm = SVC(kernel='linear')

        dirpath = os.getcwd()
        self.cvSavepath = os.path.join(dirpath, 'image', kinds + '_rfecv_3rd.jpg')
        self.fsSavepath = os.path.join(dirpath, 'image', kinds + '_rfefs_3rd.jpg')

        data1, data2, self.label = get_data(kinds)

        self.RFEtrain(data1)
        self.RFEtrain(data2)

        plt.close(1)
        plt.close(2)

    def RFEtrain(self, data):

        # # 1st
        # search_model = self._grid_search(data, self.label)
        # model = RFECV(estimator=search_model, step=1, cv=KFold(len(data)), scoring='accuracy', n_jobs=-1)
        # X = model.fit_transform(data, self.label)
        # plot_feature_selected(model, self.fsSavepath)
        # self.trainproc(X, search_model)

        # # 2nd
        # model = RFECV(estimator=self.svm, step=1, cv=KFold(len(data)), scoring='accuracy', n_jobs=-1)
        # X = model.fit_transform(data, self.label)
        # plot_feature_selected(model, self.fsSavepath)
        # self.trainproc(X, self.svm)

        # 3rd
        model = RFECV(estimator=self.svm, step=1, cv=KFold(len(data)), scoring='accuracy', n_jobs=-1)
        X = model.fit_transform(data, self.label)
        search_model = self._grid_search(X, self.label)
        plot_feature_selected(model, self.fsSavepath)
        self.trainproc(X, search_model)

    def trainproc(self, data, model):

        acc = 0
        y_label = []
        y_score = []
        for train_idx, test_idx in KFold(len(data), shuffle=True).split(data):

            model.fit(data[train_idx], self.label[train_idx])
            acc += model.score(data[test_idx], self.label[test_idx])
            y_score.append(model.decision_function(data[test_idx]))
            y_label.append(self.label[test_idx])

        print(data.shape, acc / len(data))
        plot_roc_auc(y_label, y_score, self.cvSavepath)

    def _grid_search(self, data, label):
        grid_search = GridSearchCV(self.svm, self.svm_param, n_jobs=-1, cv=KFold(len(data)))
        grid_search.fit(data, label)
        return grid_search.best_estimator_


if __name__ == "__main__":

    RFECVtrain('75n')
    RFECVtrain('75f')
    RFECVtrain('75')
    RFECVtrain('49')
