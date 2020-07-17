import numpy as np
import csv
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def getdata(path):
    with open(path, 'r') as f1:
        reader = csv.reader(f1)
        dataset = []
        for data in reader:
            dataset.append(data)
    return dataset


data1set = np.array(getdata("./dataset/data1.csv"), dtype=float)
data2set = np.array(getdata("./dataset/data2.csv"), dtype=float)
labels = np.array(getdata("./dataset/label.csv"), dtype=int).reshape(-1)


def plotroc(labelList, scoreList, acc):
    fpr, tpr, thresholds = roc_curve(labelList, scoreList)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label="area = {0:.2f}".format(roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=0.8)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.title(label="cross validation acc :{:.2f}".format(acc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.savefig("./log/test.jpg")


def plot_feature_acc(resultpath, savepath):
    featurenums = []
    accscore = []

    with open(resultpath, 'r') as f:
        for i in f:
            a = re.findall("\d+", i)
            b = re.findall("\d+\.\d+", i)
            featurenums.append(int(a[2]))
            accscore.append(float(b[1]))

    featurenums = np.array(featurenums)
    accscore = np.array(accscore)

    print(accscore.max())
    plt.plot(featurenums, accscore)
    plt.title("dataset")
    plt.xlabel("features")
    plt.ylabel("acc")
    plt.savefig(savepath, bbox_inches='tight', dpi=300)


def data1pos(idxList):
    pos = []
    for item in idxList:
        sum = 0
        for i in range(1, 117):
            sum += i
            if sum >= item:
                x = i
                y = i - 1 - (sum - item)
                break
        pos.append([x, y])
    return pos
