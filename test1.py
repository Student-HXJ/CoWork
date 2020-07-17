from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from datautils import data2set, labels
import matplotlib.pyplot as plt
import time

start = time.time()
model = SVC(kernel='linear')
rfecv = RFECV(estimator=model, step=1, cv=KFold(49), scoring='accuracy')
rfecv.fit(data2set, labels)
plt.xlabel("number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig("./log/test.jpg")
end = time.time()

print("{:.2f}s".format(end - start))
