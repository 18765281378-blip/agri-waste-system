import pandas as pd
import random
from data_loader import load_excel_data

# 加载现有路线数据
routes_df, _ = load_excel_data('农副产品加工路线+红线规则.xlsx')

# 定义用户画像取值范围
material_types = ["茶渣", "果渣（柑橘/葡萄/猕猴桃等）", "花生壳/板栗壳", "豆制品/粮食发酵副产物", "中药残余废弃物"]
mold_levels = ["无", "轻度", "严重"]
source_levels = ["明确可追溯", "基本可追溯", "来源不明"]
grade_levels = ["优", "良", "差"]
tea_storage = ["是", "否"]
fruit_browning = ["是", "否"]

# 辅助函数：判断用户等级是否在路线适用等级中
def is_grade_match(user_grade, route_grade_str):
    """判断用户的原料等级是否匹配路线的适用等级"""
    route_grade_str = str(route_grade_str)
    if user_grade in route_grade_str:
        return True
    if "（" in route_grade_str and "）" in route_grade_str:
        bracket_content = route_grade_str.split("（")[1].split("）")[0]
        if user_grade in bracket_content:
            return True
    return False

# 辅助函数：判断用户原料是否匹配路线的适用原料
def is_material_match(user_material, route_material_str):
    """判断用户的原料类型是否在路线的适用原料范围内"""
    route_material_str = str(route_material_str)
    material_keyword_map = {
        "茶渣": "茶渣",
        "果渣（柑橘/葡萄/猕猴桃等）": "柑橘皮|葡萄皮|猕猴桃|脐橙|果渣",
        "花生壳/板栗壳": "花生壳|板栗壳",
        "豆制品/粮食发酵副产物": "豆腐渣|酒糟",
        "中药残余废弃物": "五指毛桃|中药残余|布惊根|黄板树根"
    }
    keyword = material_keyword_map.get(user_material, "")
    if not keyword:
        return False
    import re
    return bool(re.search(keyword, route_material_str))

# 新增：多目标分数映射（基于路线属性）
def get_multi_scores(route):
    """
    根据路线的成本门槛、技术难度、风险等级、推荐强度，返回四个维度的分数（0-10）
    """
    cost_map = {"低": 8, "中": 5, "高": 2}
    tech_map = {"低": 8, "中": 5, "高": 2}
    carbon_map = {"低": 8, "中": 5, "高": 2}   # 风险等级越低，碳减排得分越高
    market_map = {"高": 8, "中": 5, "低": 2}   # 推荐强度越高，市场成熟度得分越高
    return {
        "成本得分": cost_map.get(route.get("成本门槛", "中"), 5),
        "技术得分": tech_map.get(route.get("技术难度", "中"), 5),
        "碳减排得分": carbon_map.get(route.get("风险等级", "中"), 5),
        "市场得分": market_map.get(route.get("推荐强度", "中"), 5),
    }

# 生成1500条模拟数据（增加多目标标签）
data = []
for _ in range(1500):
    # 随机选一条路线
    route = routes_df.sample(1).iloc[0]
    # 随机生成用户画像
    material = random.choice(material_types)
    mold = random.choice(mold_levels)
    source = random.choice(source_levels)
    grade = random.choice(grade_levels)
    
    # 专属字段（仅对应原料有值）
    tea_storage_val = random.choice(tea_storage) if material == "茶渣" else None
    fruit_browning_val = random.choice(fruit_browning) if material == "果渣（柑橘/葡萄/猕猴桃等）" else None
    
    # 核心匹配特征
    material_match = 1 if is_material_match(material, route["适用原料类"]) else 0
    grade_match = 1 if is_grade_match(grade, route["适用等级"]) else 0
    
    # 综合匹配得分（原有逻辑）
    base_score = 0
    base_score += 4 if material_match else 0
    base_score += 3 if grade_match else 0
    base_score += {"高":2, "中高":1, "中":0}.get(route["推荐强度"], 0)
    base_score += {"低":2, "中":1, "高":0}.get(route["风险等级"], 0)
    if mold == "严重":
        if "食品支线" in str(route["首选目标"]):
            base_score = 0
    label = max(0, min(12, base_score + random.uniform(-0.5, 0.5)))
    
    # 多目标分数（基于路线属性）
    multi = get_multi_scores(route)
    
    data.append({
        # 用户画像特征
        "原料类型": material,
        "霉变情况": mold,
        "来源情况": source,
        "预估等级": grade,
        "茶渣_储存时间": tea_storage_val,
        "果渣_褐变情况": fruit_browning_val,
        # 核心匹配特征
        "原料是否匹配": material_match,
        "等级是否匹配": grade_match,
        # 路线属性特征
        "路线_推荐强度": route["推荐强度"],
        "路线_风险等级": route["风险等级"],
        "路线_首选目标": route["首选目标"],
        "路线_适用原料类": route["适用原料类"],
        "路线_适用等级": route["适用等级"],
        # 标签（多目标）
        "匹配得分": label,
        "成本得分": multi["成本得分"],
        "技术得分": multi["技术得分"],
        "碳减排得分": multi["碳减排得分"],
        "市场得分": multi["市场得分"]
    })

# 保存为CSV
df = pd.DataFrame(data)
df.to_csv("recommendation_train_data.csv", index=False, encoding="utf-8-sig")
print(f"✅ 模拟训练数据已生成：recommendation_train_data.csv，共{len(df)}条样本")
print(f"✅ 样本得分分布：最小值{df['匹配得分'].min():.2f}，最大值{df['匹配得分'].max():.2f}，均值{df['匹配得分'].mean():.2f}")
print(f"✅ 多目标得分列已添加：成本得分、技术得分、碳减排得分、市场得分")