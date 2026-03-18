import pandas as pd
import numpy as np

def preprocess_features(df, is_train=True):
    """
    特征预处理：统一训练和预测时的逻辑，彻底解决NaN和特征对齐问题
    :param df: 输入的DataFrame
    :param is_train: 是否为训练阶段（训练时返回特征列名，预测时不需要）
    :return: 训练时返回 X, y, feature_cols；预测时返回 X
    """
    # 1. 填充缺失值（专属字段仅对应原料有值，其余填"无"）
    df["茶渣_储存时间"] = df["茶渣_储存时间"].fillna("无")
    df["果渣_褐变情况"] = df["果渣_褐变情况"].fillna("无")
    
    # 2. 有序分类特征的标签编码（有明确高低顺序的特征）
    ordinal_mapping = {
        "霉变情况": {"无": 0, "轻度": 1, "严重": 2},
        "来源情况": {"明确可追溯": 0, "基本可追溯": 1, "来源不明": 2},
        "预估等级": {"优": 2, "良": 1, "差": 0},
        "路线_推荐强度": {"高": 2, "中高": 1, "中": 0},
        "路线_风险等级": {"低": 2, "中": 1, "高": 0},
        "茶渣_储存时间": {"否": 0, "是": 1, "无": -1},
        "果渣_褐变情况": {"否": 0, "是": 1, "无": -1}
    }
    
    # 执行编码，避免NaN
    for feat, mapping in ordinal_mapping.items():
        if feat in df.columns:
            df[feat] = df[feat].map(mapping).fillna(-1)  # 异常值填充-1，避免NaN
    
    # 3. 无序分类特征的One-Hot编码（仅对低基数特征做，避免维度爆炸）
    one_hot_features = ["原料类型", "路线_首选目标"]
    df = pd.get_dummies(df, columns=one_hot_features, drop_first=True, dummy_na=False)
    
    # 4. 核心强特征保留（原料是否匹配、等级是否匹配，不需要编码，已经是0/1）
    core_features = ["原料是否匹配", "等级是否匹配"]
    for feat in core_features:
        if feat not in df.columns:
            raise ValueError(f"缺失核心特征：{feat}，请检查数据生成逻辑")
    
    # 5. 分离特征和标签（训练阶段）
    if is_train:
        # 移除不需要的列
        drop_cols = ["路线_适用原料类", "路线_适用等级", "匹配得分"]
        feature_cols = [col for col in df.columns if col not in drop_cols]
        X = df[feature_cols]
        y = df["匹配得分"]
        return X, y, feature_cols
    else:
        # 预测阶段，只返回特征
        drop_cols = ["路线_适用原料类", "路线_适用等级"]
        feature_cols = [col for col in df.columns if col not in drop_cols]
        return df[feature_cols]