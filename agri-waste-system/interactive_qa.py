from question_bank import question_bank

def run_qa():
    user_profile = {}
    
    print("="*60)
    print("   农副产品副产物高值化利用智能推荐系统")
    print("="*60)
    
    # 1. 通用风险评估问题
    print("\n【第一部分：通用风险评估】（共5题）")
    for q in question_bank["通用风险题"]:
        print(f"\n{q['id']}. {q['question']}")
        for i, opt in enumerate(q['options'], 1):
            print(f"   {i}. {opt}")
        
        # 优化输入容错，支持数字/文字输入
        while True:
            choice = input(f"请选择（输入数字1-{len(q['options'])}或选项文字）：").strip()
            # 数字输入
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(q['options']):
                    selected = q['options'][idx]
                    break
            # 文字模糊匹配
            else:
                matched = [opt for opt in q['options'] if choice in opt]
                if len(matched) == 1:
                    selected = matched[0]
                    break
            print(f"⚠️ 输入无效，请输入1-{len(q['options'])}的数字或选项中的文字！")
        
        user_profile[q['key']] = selected
        print(f"✅ 已选择：{selected}")
    
    # 2. 原料专属问题
    print("\n【第二部分：原料特性精准评估】")
    material_type = user_profile["原料类型"]
    
    # 原料类型与专属题库映射
    material_specific_map = {
        "茶渣": "茶渣专属题",
        "果渣（柑橘/葡萄/猕猴桃等）": "果渣专属题",
        "花生壳/板栗壳": "花生壳/板栗壳专属题",
        "豆制品/粮食发酵副产物": "豆制品/粮食发酵副产物专属题",
        "中药残余废弃物": "中药残余废弃物专属题"
    }
    
    specific_key = material_specific_map.get(material_type)
    if specific_key and specific_key in question_bank:
        specific_questions = question_bank[specific_key]
        print(f"针对「{material_type}」的补充问题（共{len(specific_questions)}题）：")
        
        for q in specific_questions:
            print(f"\n{q['id']}. {q['question']}")
            for i, opt in enumerate(q['options'], 1):
                print(f"   {i}. {opt}")
            
            # 同样的容错输入逻辑
            while True:
                choice = input(f"请选择（输入数字1-{len(q['options'])}或选项文字）：").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(q['options']):
                        selected = q['options'][idx]
                        break
                else:
                    matched = [opt for opt in q['options'] if choice in opt]
                    if len(matched) == 1:
                        selected = matched[0]
                        break
                print(f"⚠️ 输入无效，请输入1-{len(q['options'])}的数字或选项中的文字！")
            
            user_profile[q['key']] = selected
            print(f"✅ 已选择：{selected}")
    else:
        print(f"ℹ️ 暂无「{material_type}」的专属问题，直接进入推荐阶段。")
    
    print("\n✅ 信息收集完成！正在生成推荐方案...")
    return user_profile

if __name__ == "__main__":
    profile = run_qa()
    print("\n【您的原料画像】", profile)