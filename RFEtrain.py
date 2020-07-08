import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneOut
from data import data1set, data2set, labels, plotroc, getacc


def RFEprocess(model, X_train, y_tarin, X_test, features):
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    selector = RFE(model, features, step=1)
    selector = selector.fit(X_train, y_train)
    summary = np.zeros(sum(selector.support_)).tolist()
    j = 0
    k = 0
    for i in selector.support_:
        j = j + 1
        if i:
            summary[k] = j - 1
            k = k + 1
    X_train = X_train[:, summary]
    X_test = X_test[:, summary]
    return X_train, X_test


model1 = SVC(C=1, kernel="linear", probability=True)
lo = LeaveOneOut()
lo.get_n_splits(data2set)

testlabelList = []
testpredList = []
testscoreList = []
for train_idx, test_idx in lo.split(data2set):

    X_train = data2set[train_idx]
    X_test = data2set[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # RFE processing extract the feature
    RFEtrain_data, RFEtest_data = RFEprocess(model1, X_train, y_train, X_test, 100)
    model1.fit(RFEtrain_data, y_train)
    testpred = model1.predict(RFEtest_data)
    testscore = model1.decision_function(RFEtest_data)

    testpredList.append(testpred[0])
    testlabelList.append(y_test[0])
    testscoreList.append(testscore[0])

getacc(testlabelList, testpredList)
plotroc(testlabelList, testscoreList)
