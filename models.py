"""四种模型的候选参数，以及 MLP 的逐 mini-batch 训练"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SEED = 42


def get_experiments():
    """返回 (模型名, 参数字典)"""
    configs = []
    for alpha in [0.01, 0.1, 1, 10]:
        configs.append(("Naive Bayes", {"alpha": alpha}))
    for c in [0.1, 1, 10, 100]:
        configs.append(("SVM", {"kernel": "linear", "C": c}))
    for c in [0.01, 0.1, 1, 10, 100]:
        configs.append(("Logistic Regression", {"C": c}))
    # MLP：只调隐藏层结构，训练轮数和 batch_size 保持一致
    for layers in [(50,), (100,), (100, 50)]:
        configs.append(("MLP", {"hidden_layer_sizes": layers, "epochs": 30, "batch_size": 128}))
    return configs


def create_model(name, params):
    """统一创建模型；MLP 的 epochs、batch_size 由训练函数负责"""
    if name == "Naive Bayes":
        return MultinomialNB(alpha=params["alpha"])
    if name == "SVM":
        return SVC(kernel=params["kernel"], C=params["C"], random_state=SEED)
    if name == "Logistic Regression":
        return LogisticRegression(C=params["C"], max_iter=1000, random_state=SEED)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"],
                             solver="adam", random_state=SEED)
    raise ValueError(f"未知模型: {name}")


def train_mlp(model, X_train, y_train, epochs=10, batch_size=128):
    """每次 partial_fit 使用一个 mini-batch，并记录一次 sklearn loss_"""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    rng = np.random.RandomState(SEED)
    losses = []
    for epoch in range(epochs):
        indices = rng.permutation(len(y_train))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            if epoch == 0 and start == 0:
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)
            losses.append(float(model.loss_))
    return losses


def train_model(name, params, X_train, y_train):
    """返回训练好的模型和损失历史（非 MLP 为 None）"""
    model = create_model(name, params)
    if name == "MLP":
        losses = train_mlp(model, X_train, y_train,
                           epochs=params["epochs"], batch_size=params["batch_size"])
    else:
        model.fit(X_train, y_train)
        losses = None
    return model, losses
