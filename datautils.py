import re
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def noiseadd(signal):
    SNR = 5
    noise = np.random.randn(signal.shape[0], signal.shape[1])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal)**2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise


def GPUs(x):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        tf.config.set_visible_devices([], 'GPU')
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[x], 'GPU')


def plot_roc_auc(labelList, scoreList, savepath):
    fpr, tpr, thresholds = roc_curve(labelList, scoreList)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label="area = {0:.2f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=0.8)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.title(label="cross validation")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.savefig(savepath)


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


def getacc(arr):
    accset = []
    for i in range(49):
        accset.append(arr[i][0])
    acc = np.sum(accset) / len(accset)
    return acc


def getroc(arr, savepath):
    scoreset = []
    labelset = []
    for i in range(49):
        scoreset.append(arr[i][1])
        labelset.append(arr[i][2])

    plot_roc_auc(labelset, scoreset, savepath)


def negtozero(labels):
    tmp = []
    for item in labels:
        if (item == -1):
            item = 0
        tmp.append(item)
    labels = np.array(tmp, dtype=int).reshape(-1)
    return labels


def getdata(path):
    with open(path, 'r') as f1:
        reader = csv.reader(f1)
        dataset = []
        for data in reader:
            dataset.append(data)
    return dataset
