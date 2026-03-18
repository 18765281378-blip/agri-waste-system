import pandas as pd

def load_excel_data(file_name='农副产品加工路线+红线规则.xlsx'):
    """
    读取Excel数据，使用相对路径
    :param file_name: Excel文件名（默认与代码在同一目录）
    :return: 工艺路线表和红线规则表的DataFrame
    """
    try:
        # 读取表1：工艺路线库
        routes_df = pd.read_excel(file_name, sheet_name='表1')
        # 读取表2：红线规则
        rules_df = pd.read_excel(file_name, sheet_name='表2')
        
        # 简单清洗：去除空行、重置索引
        routes_df = routes_df.dropna(how='all').reset_index(drop=True)
        rules_df = rules_df.dropna(how='all').reset_index(drop=True)
        
        print(f"✅ 成功读取数据：\n   - 工艺路线：{len(routes_df)} 条\n   - 红线规则：{len(rules_df)} 条")
        return routes_df, rules_df
    
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 '{file_name}'，请确保文件与代码在同一目录！")
        return None, None
    except Exception as e:
        print(f"❌ 读取文件时出错：{e}")
        return None, None

# 测试读取
if __name__ == "__main__":
    routes, rules = load_excel_data()
    if routes is not None:
        print("\n工艺路线表列名：", routes.columns.tolist())
        print("\n红线规则表列名：", rules.columns.tolist())