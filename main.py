import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from models import get_experiments, train_model, SEED
from plots import plot_mlp_loss


def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].to_numpy()
    X_test = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test


def main():
    X_train, y_train, X_test = load_data()
    print(f"数据加载成功！训练集样本 {len(X_train)} 条，测试集样本 {len(X_test)} 条")

    # 先划分，再 TF-IDF，防止验证集信息泄露
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_split)
    X_val_tfidf = vectorizer.transform(X_val)
    print(f"TF-IDF 完成！训练集/验证集: {len(X_train_split)}/{len(X_val)}; TF-IDF 特征数: {X_train_tfidf.shape[1]}")

    # 模型学习流程：训练 -> 预测 -> 计算准确率 -> 记录实验结果
    results = []
    best_mlp_loss = None
    best_mlp_accuracy = -1.0
    best_mlp_params = None
    for name, params in get_experiments():
        print(f"训练 {name}: {params}", flush=True)
        model, losses = train_model(name, params, X_train_tfidf, y_train_split)
        accuracy = accuracy_score(y_val, model.predict(X_val_tfidf))
        results.append({"Model": name, "Parameters": str(params), "Accuracy": accuracy})
        print(f"  Validation Accuracy: {accuracy:.4f}")
        if name == "MLP" and accuracy > best_mlp_accuracy:
            best_mlp_accuracy = accuracy
            best_mlp_loss = losses
            best_mlp_params = params.copy()

    # 汇总所有模型对Accuracy进行排名
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print("\n--- 模型验证结果（按 Accuracy 排序） ---")
    print(results_df.to_string(index=False))

    # 只绘制验证 Accuracy 最好的 MLP 配置对应的原始损失记录
    if best_mlp_loss is not None:
        plot_mlp_loss(best_mlp_loss, epochs=best_mlp_params["epochs"])
        print("\n已保存 MLP 原始 Mini-batch Loss 曲线")

    # 选择最终模型LR：综合考虑验证集性能和模型复杂度
    selected_name = "Logistic Regression"
    
    # 从实验结果中找到该模型验证准确率最高的一组
    selected_result = (results_df[results_df["Model"] == selected_name].iloc[0])
    
    # 从实验配置中找回对应的参数字典
    selected_params = next(params for name, params in get_experiments() if name == selected_name and str(params) == selected_result["Parameters"])
    
    # 使用全部有标签数据重新拟合 TF-IDF
    final_vectorizer = TfidfVectorizer(max_features=5000)
    X_full_tfidf = final_vectorizer.fit_transform(X_train)
    X_test_tfidf = final_vectorizer.transform(X_test)
    
    # 重新训练选定模型
    final_model, _ = train_model(selected_name, selected_params, X_full_tfidf, y_train)
    
    # 预测并保存结果
    predictions = final_model.predict(X_test_tfidf)
    pd.DataFrame({"target": predictions}).to_csv("predictions.csv", index=False, header=False)
    
    print(f"\n最终选择模型: {selected_name}")
    print(f"超参数: {selected_params}")
    print(f"验证集 Accuracy: {selected_result['Accuracy']:.4f}")
    print("预测成功！结果已保存到: ./predictions.csv")
    

if __name__ == "__main__":
    main()
