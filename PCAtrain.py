import os
import sys
sys.path.append(os.getcwd())
from datautils import *


def PCAprocess(X_train, X_test):
    pca = PCA(n_components=48).fit(X_train)
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test


model1 = SVC(C=1, kernel="linear", probability=True)
lo = LeaveOneOut()
lo.get_n_splits(data2set)

testpredList = []
testlabelList = []
testscoreList = []
for train_idx, test_idx in KFold(n_splits=49).split(data2set):

    X_train = data2set[train_idx]
    X_test = data2set[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # RFE processing extract the feature
    PCAtrain_data, PCAtest_data = PCAprocess(X_train, X_test)
    model1.fit(PCAtrain_data, y_train)
    testpred = model1.predict(PCAtest_data)
    testscore = model1.decision_function(PCAtest_data)

    testpredList.append(testpred[0])
    testlabelList.append(y_test[0])
    testscoreList.append(testscore[0])

plotroc(testlabelList, testscoreList)
