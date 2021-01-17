from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

bg_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,  # 子模型个数
    max_samples=100,  # 模型可见样本数量
    bootstrap=True,  # 可放回抽样
    n_jobs=-1,
    max_features=1,  # 每个样本可见特征数量
    bootstrap_features=True,  # 可放回抽样
    oob_score=True,
)

rf_clf = RandomForestClassifier()
