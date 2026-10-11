# 实验一：新闻文本分类

## 文件说明
- `main.py`：数据加载、划分、TF-IDF、四模型统一比较、选择最优配置并预测
- `models.py`：Naive Bayes、SVM、Logistic Regression、MLP 候选参数与训练
- `plots.py`：MLP 模型的原始 Mini-batch Loss 曲线绘制
- `test_data_unlabeled.csv`和`train_data.csv`：数据

## 运行

1. 安装依赖：`pip install numpy pandas scikit-learn matplotlib`
2. 在该目录运行：`python main.py`


## 实验设计
- 划分：训练集 80%，验证集 20%，按标签分层，随机种子 42
- TF-IDF：仅在当前训练子集上拟合，`max_features=5000`
- 调参：Naive Bayes 比较 alpha；SVM比较C（正则化参数）；LR 比较C（正则化强度的倒数）；MLP 比较 `(50,)`、`(100,)`、`(100, 50)` 三种隐藏层结构，固定 epochs=30、batch_size=128
- 选择：Logistic Regression；固定最优参数后，重新用全部标注训练数据拟合 TF-IDF 和模型，预测无标签测试集

## 注意
1. 确保当前目录下有`test_data_unlabeled.csv`和`train_data.csv`
2. 该程序运行后，会在当前目录有以下输出结果： `mlp_loss_curve.png`（损失函数曲线）、`predictions.csv`（预测结果）。该仓库为原始代码，当前并不包括这两个输出文件
3. 由于该程序中有多个模型的多个参数设置，所以运行时间较长