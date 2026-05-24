# 导入依赖库
import lightgbm as lgb
import matplotlib
# 🔥 关键：使用 Agg 后端，不弹出窗口，直接保存高清图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 核心配置 --------------------------
feature_names = [
    "原料类型", "原料等级", "含水率", "膳食纤维含量",
    "多酚含量", "风险等级", "杂质率", "利用优先级"
]

# -------------------------- 模拟模型（替换为你的真实模型） --------------------------
X = np.random.rand(1000, 8)
y = np.random.randint(0, 3, 1000)
train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1}
model = lgb.train(params, train_data, num_boost_round=50)

# -------------------------- 绿色主题可视化（带数值标注） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 7)  # 稍微加宽一点，方便放数字

# 配色
MAIN_GREEN = "#2E7D32"    # 深绿
LIGHT_GREEN = "#81C784"  # 浅绿

# 获取数据并排序
importance = model.feature_importance()
sorted_idx = np.argsort(importance)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = importance[sorted_idx]

# 创建画布
fig, ax = plt.subplots()

# 绘制横向柱状图
bars = ax.barh(sorted_features, sorted_importance, 
                color=[MAIN_GREEN if i > len(sorted_features)/2 else LIGHT_GREEN for i in range(len(sorted_features))],
                edgecolor='white', linewidth=1.5)

# 🔥 核心新增：在每个柱子末端标注具体数值
for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
    # 把数字写在柱子末端，稍微右移一点
    ax.text(value + 5, bar.get_y() + bar.get_height()/2, 
            f'{value}', 
            ha='left', va='center', 
            fontsize=11, fontweight='bold', color=MAIN_GREEN)

# 美化样式
ax.set_title('LightGBM 模型特征贡献度可视化', fontsize=16, fontweight='bold', color=MAIN_GREEN, pad=25)
ax.set_xlabel('特征贡献度', fontsize=12, color='#333333')
ax.set_ylabel('核心特征', fontsize=12, color='#333333')

# 调整坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.3, color='#666666')
ax.set_axisbelow(True)

# 确保布局紧凑，高清保存
plt.tight_layout()
plt.savefig("模型特征贡献_带数值版.png", dpi=300, bbox_inches='tight')
print("✅ 带数值的高清图片已保存！")