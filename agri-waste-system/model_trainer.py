import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from feature_engineering import preprocess_features

# 1. 加载数据
df = pd.read_csv("recommendation_train_data.csv", encoding="utf-8-sig")

# 2. 特征预处理
X, y, feature_cols = preprocess_features(df, is_train=True)
print(f"✅ 特征预处理完成，共{len(feature_cols)}个特征")
print(f"✅ 特征列表：{feature_cols}")  

# 3. 划分训练/测试集（8:2，固定随机种子保证可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 4. 优化LightGBM模型参数，适配回归任务
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

# 5. 定义回调函数
callbacks = [
    early_stopping(stopping_rounds=15),
    log_evaluation(period=10)
]

# 6. 训练模型
print("\n🚀 开始训练模型...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)

# 7. 模型评估
print("\n📊 模型评估结果：")
# 用最优迭代次数预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"   平均绝对误差（MAE）：{mae:.2f}（越小越好，理想<0.3）")
print(f"   决定系数（R²）：{r2:.2f}（越接近1越好，理想>0.9）")

# 8. 打印特征重要性（验证核心特征是否生效）
print("\n🏆 特征重要性Top5：")
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)
print(feature_importance.head().to_string(index=False))

# 9. 保存模型、特征列名（必须和特征工程严格对齐）
joblib.dump(model, "recommendation_model.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")
print("\n✅ 模型已保存：recommendation_model.pkl")
print("✅ 特征列名已保存：feature_columns.pkl")