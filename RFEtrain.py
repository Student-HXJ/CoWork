import os
import sys

sys.path.append(os.getcwd())
from datautils import *


class RFEtrain:

    def __init__(self, dataset, labels):

        # model
        self.model = SVC(C=1, kernel="linear")
        # dataset
        self.dataset = dataset
        self.labels = labels
        self.samples = dataset.shape[0]
        self.features = dataset.shape[1]

    def useRFECV(self, savepath):

        rfecv = RFECV(estimator=self.model, step=1, cv=KFold(49), scoring='accuracy')
        rfecv.fit(self.dataset, self.labels)

        plt.xlabel("number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.savefig(savepath)
        for i in range(len(rfecv.grid_scores_)):
            print(i, rfecv.grid_scores_[i])

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
        score = self.model.decision_function(data[test_idx][:, summary])
        acc = accuracy_score(label[test_idx], pred)

        result = np.array([acc, score, label[test_idx]])
        return result

    def cross_validation(self, features):
        result = Parallel(16)(delayed(self.RFEprocess)(
            self.dataset,
            self.labels,
            train_idx,
            test_idx,
            features,
        ) for train_idx, test_idx in KFold(n_splits=self.samples).split(self.dataset))

        acc = getacc(result)
        # getroc(result)
        return acc

    def feature_select(self):
        for i in range(100, self.features + 1, 5):
            start = time.time()
            acc = self.cross_validation(i)
            end = time.time()
            print("time: {:.2f}s, featurenum: {:d}, acc: {:.2f}".format(end - start, i, acc))


if __name__ == "__main__":
    RFEtrain(data1set, labels).useRFECV("./log/result1.jpg")
    # plot_feature_acc("./log/result1.log", "./log/result1.jpg")
