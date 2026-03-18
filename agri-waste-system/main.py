from data_loader import load_excel_data
from interactive_qa import run_qa
from recommender import RouteRecommender
import json

def main():
    # 1. 加载数据（使用相对路径）
    print("📂 正在加载工艺路线库...")
    routes_df, rules_df = load_excel_data('农副产品加工路线+红线规则.xlsx')
    
    if routes_df is None or rules_df is None:
        return
    
    # 2. 运行问答收集用户画像
    user_profile = run_qa()
    
    # 3. 初始化推荐器并生成推荐
    print("\n🔍 正在分析并生成推荐方案...")
    recommender = RouteRecommender(routes_df, rules_df)
    top3_recommendations = recommender.recommend_top3(user_profile)
    
    if top3_recommendations is None:
        return
    
    # 4. 输出推荐结果
    print("\n" + "="*70)
    print("   【智能推荐结果】Top3 最优工艺路线")
    print("="*70)
    
    for idx, (_, row) in enumerate(top3_recommendations.iterrows(), 1):
        print(f"\n🏆 第 {idx} 名（匹配得分：{row['匹配得分']}/12）")
        print(f"   路线编号：{row['路线编号']}")
        print(f"   路线名称：{row['路线名称']}")
        print(f"   适用原料：{row['适用原料类']}")
        print(f"   首选目标：{row['首选目标']}")
        print(f"   推荐强度：{row['推荐强度']} | 风险等级：{row['风险等级']}")
        print(f"   推荐理由：{row['推荐理由关键词']}")
    
    # 5. 保存结果为JSON（为板块三做准备）
    output_data = {
        "user_profile": user_profile,
        "top3_recommendations": top3_recommendations.to_dict(orient='records')
    }
    
    with open('qa_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("\n💾 结果已保存至 'qa_result.json'，可用于后续网站开发！")

if __name__ == "__main__":
    main()
    