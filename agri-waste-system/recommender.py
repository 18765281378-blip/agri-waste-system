import pandas as pd
import joblib
import re

class RouteRecommender:
    def __init__(self, routes_df, rules_df):
        self.routes = routes_df
        self.rules = rules_df
        # 加载训练好的模型和特征列名
        self.model = joblib.load("recommendation_model.pkl")
        self.train_feature_cols = joblib.load("feature_columns.pkl")
        print("✅ 推荐器初始化完成，模型和特征加载成功")
    
    @staticmethod
    def normalize_user_profile(user_profile: dict) -> dict:
        """
        归一化用户输入，增强网站/API容错。
        - 等级支持「中」，临时映射为「良」以兼容既有模型（训练集只有优/良/差）
        """
        profile = dict(user_profile or {})
        if profile.get("预估等级") == "中":
            profile["预估等级"] = "良"
        return profile
    
    # 复用和训练时一致的辅助函数，保证逻辑统一
    @staticmethod
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
    
    @staticmethod
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
        return bool(re.search(keyword, route_material_str))
    
    def check_red_line_reasons(self, user_profile, route) -> list[str]:
        """单条路线红线规则校验，返回触发原因列表（空列表表示合规）"""
        reasons: list[str] = []
        # 1. 霉变严重，禁止食品支线
        if user_profile.get("霉变情况") == "严重":
            if "食品支线" in str(route["首选目标"]):
                reasons.append("霉变严重：禁用食品支线")
        # 2. 来源不明，禁止高附加值精细提取路线
        if user_profile.get("来源情况") == "来源不明":
            if "活性成分提取" in str(route["路线名称"]) or "天然色素" in str(route["路线名称"]) or "多酚" in str(route["路线名称"]):
                reasons.append("来源不明：禁用精细提取/高附加值路线")
        # 3. 等级为差，禁止优/良限定的高要求路线
        if user_profile.get("预估等级") == "差":
            route_grade = str(route["适用等级"])
            if "差" not in route_grade and "优/良" in route_grade:
                reasons.append("原料等级为差：禁用仅限优/良的高要求路线")
        # 4. 茶渣专属规则：含调味/储存超3天，禁止活性成分提取
        if user_profile.get("原料类型") == "茶渣":
            if user_profile.get("是否含调味") == "是" or user_profile.get("储存时间") == "是":
                if "活性成分提取" in str(route["路线名称"]):
                    reasons.append("茶渣含调味或存放过久：禁用活性成分提取")
        # 5. 果渣专属规则：褐变严重，禁止天然色素提取
        if user_profile.get("原料类型") == "果渣（柑橘/葡萄/猕猴桃等）" and user_profile.get("褐变情况") == "是":
            if "天然色素" in str(route["路线名称"]):
                reasons.append("果渣褐变明显：禁用天然色素提取")
        
        return reasons
    
    def apply_red_line_rules(self, user_profile, route):
        """兼容旧调用：返回是否禁用"""
        return len(self.check_red_line_reasons(user_profile, route)) > 0
    
    def preprocess_single_route(self, route, user_profile):
        """预处理单条路线，生成和训练时完全一致的特征"""
        # 1. 计算核心匹配特征（和训练时完全一致）
        material_match = 1 if self.is_material_match(user_profile["原料类型"], route["适用原料类"]) else 0
        grade_match = 1 if self.is_grade_match(user_profile["预估等级"], route["适用等级"]) else 0
        
        # 2. 构造和训练时结构完全一致的样本
        sample = {
            # 用户画像特征
            "原料类型": user_profile["原料类型"],
            "霉变情况": user_profile["霉变情况"],
            "来源情况": user_profile["来源情况"],
            "预估等级": user_profile["预估等级"],
            "茶渣_储存时间": user_profile.get("储存时间", None),
            "果渣_褐变情况": user_profile.get("褐变情况", None),
            # 核心匹配特征
            "原料是否匹配": material_match,
            "等级是否匹配": grade_match,
            # 路线属性特征
            "路线_推荐强度": route["推荐强度"],
            "路线_风险等级": route["风险等级"],
            "路线_首选目标": route["首选目标"],
            "路线_适用原料类": route["适用原料类"],
            "路线_适用等级": route["适用等级"],
        }
        
        # 3. 转为DataFrame
        sample_df = pd.DataFrame([sample])
        
        # 4. 复用训练时的特征工程逻辑（is_train=False，预测模式）
        from feature_engineering import preprocess_features
        processed_df = preprocess_features(sample_df, is_train=False)
        
        # 5. 严格对齐训练时的特征列，缺失的列补0
        for col in self.train_feature_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
        # 只保留训练时用到的特征列，保证顺序一致
        processed_df = processed_df[self.train_feature_cols]
        
        return processed_df
    
    def calculate_route_score(self, route, user_profile):
        """用训练好的模型计算单条路线的匹配得分"""
        # 预处理特征
        X = self.preprocess_single_route(route, user_profile)
        # 模型预测得分
        score = self.model.predict(X)[0]
        # 限制得分范围在0-12之间
        score = max(0, min(12, score))
        return score

    def explain_route(self, route, user_profile):
        """
        生成可解释信息：
        - 红线触发原因（若有）
        - 核心匹配（原料/等级）
        - 模型贡献度Top要点（若模型支持 pred_contrib）
        """
        profile = self.normalize_user_profile(user_profile)
        red_line_reasons = self.check_red_line_reasons(profile, route)

        material_match = self.is_material_match(profile.get("原料类型"), route.get("适用原料类"))
        grade_match = self.is_grade_match(profile.get("预估等级"), route.get("适用等级"))
        core_reasons = []
        core_reasons.append("原料匹配" if material_match else "原料不匹配")
        core_reasons.append("等级匹配" if grade_match else "等级不匹配")

        contrib_top = []
        try:
            X = self.preprocess_single_route(route, profile)
            contrib = self.model.predict(X, pred_contrib=True)
            # shape: (1, n_features+1) 最后一列是bias
            values = contrib[0]
            feature_names = list(self.train_feature_cols) + ["(bias)"]
            pairs = list(zip(feature_names, values))
            pairs = [p for p in pairs if p[0] != "(bias)"]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, val in pairs[:5]:
                contrib_top.append({"feature": name, "contribution": float(val)})
        except Exception:
            contrib_top = []

        return {
            "red_line_reasons": red_line_reasons,
            "core_reasons": core_reasons,
            "feature_contributions_top": contrib_top,
        }
    
    def recommend_top3(self, user_profile):
        """核心推荐流程：过滤→打分→排序→Top3"""
        user_profile = self.normalize_user_profile(user_profile)
        print(f"\n🔍 正在为您的原料筛选匹配路线...")
        # 第一步：先做粗过滤，只保留原料匹配的路线（大幅减少无效计算）
        candidate_routes = self.routes[
            self.routes.apply(
                lambda x: self.is_material_match(user_profile["原料类型"], x["适用原料类"]),
                axis=1
            )
        ].copy()
        print(f"   第一步：原料匹配过滤，剩余{len(candidate_routes)}条候选路线")
        
        # 第二步：红线规则过滤，移除禁用路线
        forbidden_index = []
        for idx, route in candidate_routes.iterrows():
            if self.apply_red_line_rules(user_profile, route):
                forbidden_index.append(idx)
        candidate_routes = candidate_routes.drop(forbidden_index)
        print(f"   第二步：红线规则过滤，剩余{len(candidate_routes)}条合规路线")
        
        # 校验：无可用路线
        if candidate_routes.empty:
            print("❌ 无可用的合规路线，请检查原料情况！")
            return None
        
        # 第三步：模型计算每条路线的匹配得分
        print("   第三步：模型计算匹配得分...")
        candidate_routes["匹配得分"] = candidate_routes.apply(
            lambda x: self.calculate_route_score(x, user_profile), axis=1
        )
        
        # 第四步：按得分降序排序，取Top3
        top3 = candidate_routes.sort_values(by="匹配得分", ascending=False).head(3)
        
        # 第五步：选择关键列返回
        result_cols = [
            "路线编号",
            "路线名称",
            "适用原料类",
            "适用等级",
            "首选目标",
            "备选目标",
            "成本门槛",
            "技术难度",
            "风险等级",
            "推荐强度",
            "质量要求",
            "来源要求",
            "添加物限制",
            "排除条件",
            "推荐理由关键词",
            "禁忌提示",
            "替代路线",
            "匹配得分",
        ]
        existing = [c for c in result_cols if c in top3.columns]
        return top3[existing]