import os
import numpy as np
import random
import tensorflow as tf
from CoWork.datautils import GPUs, negtozero
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold

np.random.seed(2021)
random.seed(2021)
tf.random.set_seed((2021))
GPUs(0)

dirpath = os.getcwd()
datapath = os.path.join(dirpath, 'CoWork', 'dataset', '75')
data1set = np.load(os.path.join(datapath, 'data1.npy'))
data2set = np.load(os.path.join(datapath, 'data2.npy'))
labels = np.load(os.path.join(datapath, 'label.npy'))

savepath = os.path.join(dirpath, 'CoWork', 'log', 'roc.jpg')
features = [116, 55]


def svm(x_train, y_train, x_test, y_test):
    model = SVC(C=1, kernel='linear', probability=True)
    model.fit(x_train, y_train)
    pred_train = model.predict_proba(x_train)
    pred_test = model.predict_proba(x_test)
    return pred_train, pred_test


def RFEproc(x_train, y_train, x_test, y_test, features):
    selector = RFE(SVC(C=1, kernel='linear'), n_features_to_select=features)
    selector = selector.fit(x_train, y_train)

    summary = np.zeros(sum(selector.support_)).tolist()
    j = 0
    k = 0
    for i in selector.support_:
        j = j + 1
        if i:
            summary[k] = j - 1
            k = k + 1
    return x_train[:, summary], x_test[:, summary]


def ModelInput(op):
    if (op == 'raw'):
        # 原始输入
        img = tf.keras.Input(shape=(6670))
        itt = tf.keras.Input(shape=(377))
    elif (op == 'svm'):
        # svm 预处理输入
        img = tf.keras.Input(shape=(2))
        itt = tf.keras.Input(shape=(2))
    elif (op == 'rfe'):
        # RFE 预处理输入
        img = tf.keras.Input(shape=(116))
        itt = tf.keras.Input(shape=(55))

    return img, itt


def DataInput(train_idx, test_idx, op):
    x1_train, x1_test = data1set[train_idx], data1set[test_idx]
    x2_train, x2_test = data2set[train_idx], data2set[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    if op == 'raw':
        x1_train_, x1_test_ = x1_train, x1_test
        x2_train_, x2_test_ = x2_train, x2_test

    elif op == 'svm':
        # svm 预处理
        x1_train_, x1_test_ = svm(x1_train, y_train, x1_test, y_test)
        x2_train_, x2_test_ = svm(x2_train, y_train, x2_test, y_test)

    elif op == 'rfe':
        # RFE 预处理
        x1_train_, x1_test_ = RFEproc(x1_train, y_train, x1_test, y_test, features[0])
        x2_train_, x2_test_ = RFEproc(x2_train, y_train, x2_test, y_test, features[1])

    return x1_train_, x2_train_, x1_test_, x2_test_, y_train, y_test


def res(x):
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    return x


def mymodel(op):
    img_raw, itt_raw = ModelInput(op)

    if op == 'rfe':
        img = tf.keras.layers.Dense(55, activation='relu')(img_raw)
        itt = tf.keras.layers.Dense(8, activation='relu')(itt_raw)

    elif op == 'svm':
        img = tf.keras.layers.Dense(2, activation='relu')(img_raw)
        itt = tf.keras.layers.Dense(8, activation='relu')(itt_raw)

    elif op == 'raw':
        img = tf.keras.layers.Dense(377, activation='relu')(img_raw)
        itt = tf.keras.layers.Dense(8, activation='relu')(itt_raw)

    ec = tf.keras.layers.Subtract()([img, itt_raw])

    concat = tf.keras.layers.Concatenate(axis=-1)([itt_raw, ec])

    ef = res(concat)

    add = tf.keras.layers.Add()([itt, ef])

    output = tf.keras.layers.Dense(2, activation='softmax')(add)

    model = tf.keras.Model(inputs=[img_raw, itt_raw], outputs=output, name=op)
    return model


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def train_model(op):
    # y_score = []
    # y_true = []
    accs = 0
    cnt = 0
    for train_idx, test_idx in KFold(n_splits=49).split(labels):
        x1_train, x2_train, x1_test, x2_test, y_train, y_test = DataInput(train_idx, test_idx, op)
        y_train = negtozero(y_train)
        y_test = negtozero(y_test)
        model = mymodel(op)

        # sparse_categorical_crossentropy  binary_crossentropy  hinge squared_hinge
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # 3000 - 65%
        model.fit(x=[x1_train, x2_train], y=y_train, epochs=3000, verbose=0)

        # # test
        pred = model.predict([x1_test, x2_test])
        acc = accuracy(pred, y_test)

        accs += acc.numpy()
        cnt += 1

        # y_score.append(pred[:, 1])
        # y_true.append(labels[test_idx])

        del model

    acc = accs / cnt
    print("{:.2%}".format(acc))
    # plot_roc_auc(y_true, y_score, savepath)


if __name__ == "__main__":
    op = 'rfe'
    mymodel(op).summary()
    train_model(op)
    print("done!")
