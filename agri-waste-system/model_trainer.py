import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from feature_engineering import preprocess_features

# 1. 加载数据
df = pd.read_csv("recommendation_train_data.csv", encoding="utf-8-sig")

# 2. 特征预处理（获取特征矩阵 X，并返回特征列名）
X, _, feature_cols = preprocess_features(df, is_train=True)
print(f"✅ 特征预处理完成，共 {len(feature_cols)} 个特征")
print(f"✅ 特征列表（前10个）：{feature_cols[:10]} ...")
print(f"✅ 特征总数：{len(feature_cols)}")

# 将特征列表保存到文本文件，便于检查
with open("feature_columns_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))
print("✅ 特征列表已保存至 feature_columns_list.txt")

# 目标列（五维：匹配得分 + 四个多目标得分）
target_cols = ["匹配得分", "成本得分", "技术得分", "碳减排得分", "市场得分"]

models = {}
results = {}

for target in target_cols:
    print(f"\n🚀 训练模型：{target}")
    y = df[target]
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 模型参数
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbosity=1,
        objective="regression",
        metric="mae"
    )
    
    callbacks = [
        early_stopping(stopping_rounds=15),
        log_evaluation(period=10)
    ]
    
    print(f"   开始训练 {target} 模型...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks
    )
    
    # 评估
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   {target} - MAE: {mae:.2f}, R²: {r2:.2f}, 最优迭代: {model.best_iteration_}")
    
    models[target] = model
    results[target] = {"mae": mae, "r2": r2, "best_iteration": model.best_iteration_}

# 删除旧的单模型文件（如果存在），避免加载错误
old_model_path = "recommendation_model.pkl"
if os.path.exists(old_model_path):
    os.remove(old_model_path)
    print(f"✅ 已删除旧的单模型文件：{old_model_path}")

# 保存所有多目标模型
joblib.dump(models, "recommendation_models.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")
print("\n✅ 所有模型已保存：recommendation_models.pkl")
print("✅ 特征列名已保存：feature_columns.pkl")

# 保存评估结果到文本文件
with open("model_evaluation.txt", "w", encoding="utf-8") as f:
    f.write("===== 多目标模型评估结果 =====\n")
    for target, res in results.items():
        f.write(f"\n{target}:\n")
        f.write(f"  MAE: {res['mae']:.4f}\n")
        f.write(f"  R²: {res['r2']:.4f}\n")
        f.write(f"  最优迭代次数: {res['best_iteration']}\n")
print("\n✅ 评估结果已保存至 model_evaluation.txt")

# 提示用户下一步操作
print("\n" + "="*60)
print("⚠️  重要提示：")
print("   1. 请确保已将 `recommender.py` 中的模型加载改为多目标模型（`recommendation_models.pkl`）。")
print("   2. 如果存在旧的 `recommendation_model.pkl`，系统已自动删除。")
print("   3. 重新启动 FastAPI 服务后，特征数不匹配的错误应已解决。")
print("="*60)