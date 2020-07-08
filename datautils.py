import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

data1path = "/Users/hxj/Desktop/data1.csv"
data2path = "/Users/hxj/Desktop/data2.csv"
labelpath = "/Users/hxj/Desktop/label.csv"
with open(labelpath, 'r') as f3:
    labelreader = csv.reader(f3)
    labels = []
    for label in labelreader:
        label = int(label[0])
        if label == -1:
            labels.append(0)
        else:
            labels.append(1)

with open(data2path, 'r') as f2:
    data2reader = csv.reader(f2)
    data2set = []
    for data in data2reader:
        data2set.append(data)

with open(data1path, 'r') as f1:
    data1reader = csv.reader(f1)
    data1set = []
    for data in data1reader:
        data1set.append(data)

data1set = np.array(data1set)
data2set = np.array(data2set)
labels = np.array(labels, dtype=int)


def getacc(labelList, predList):
    acc = accuracy_score(labelList, predList)
    print(acc)


def plotroc(labelList, scoreList):
    fpr, tpr, thresholds = roc_curve(labelList, scoreList)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label="area = {0:.2f}".format(roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=0.8)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.show()
