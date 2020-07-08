from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from data import data1set, data2set, labels, plotroc, getacc

model1 = SVC(C=1, kernel='linear')
lo = LeaveOneOut()
lo.get_n_splits(data2set)

trainlabelList = []
testlabelList = []
trainpredList = []
testpredList = []
trainscoreList = []
testscoreList = []

for train_idx, test_idx in lo.split(data2set):

    X_train = data2set[train_idx]
    X_test = data2set[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    model1.fit(X_train, y_train)
    testscore = model1.decision_function(X_test)
    testpred = model1.predict(X_test)
    testlabelList.append(y_test[0])
    testpredList.append(testpred[0])
    testscoreList.append(testscore[0])

getacc(testlabelList, testpredList)
plotroc(testlabelList, testscoreList)
