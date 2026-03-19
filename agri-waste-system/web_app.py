from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from data_loader import load_excel_data
from question_bank import question_bank
from recommender import RouteRecommender


ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "web" / "templates"
STATIC_DIR = ROOT / "web" / "static"


app = FastAPI(title="副产物处置方案智能推荐系统", version="0.1.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RecommendRequest(BaseModel):
    user_profile: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def _startup() -> None:
    routes_df, rules_df = load_excel_data(str(ROOT / "农副产品加工路线+红线规则.xlsx"))
    if routes_df is None or rules_df is None:
        raise RuntimeError("Excel 数据加载失败，请检查文件是否存在且可读")
    app.state.routes_df = routes_df
    app.state.rules_df = rules_df
    app.state.recommender = RouteRecommender(routes_df, rules_df)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "工艺知识库 + 数据驱动推荐",
        },
    )


@app.get("/grading", response_class=HTMLResponse)
def grading(request: Request):
    grading_dir = STATIC_DIR / "grading"
    materials_dict = {}  # 用于按原料分组

    if grading_dir.exists() and grading_dir.is_dir():
        for fp in sorted(grading_dir.iterdir()):
            if fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                # 从文件名解析原料和等级（格式：原料_等级.jpg）
                stem = fp.stem
                parts = stem.split('_')
                if len(parts) >= 2:
                    material = parts[0]
                    grade = parts[1]
                else:
                    # 如果文件名不符合规则，跳过或给默认值
                    continue

                # 为不同等级设置标签颜色（用于前端）
                grade_color = {
                    "优": "#10b981",  # 绿色
                    "良": "#3b82f6",  # 蓝色
                    "中": "#f59e0b",  # 橙色
                    "差": "#ef4444"   # 红色
                }.get(grade, "#6b7280")

                # 获取文字描述（调用下面的辅助函数）
                description = get_grade_description(material, grade)
                suggestion = get_grade_suggestion(grade)

                # 构建图片信息
                img_info = {
                    "filename": fp.name,
                    "url": f"/static/grading/{fp.name}",
                    "material": material,
                    "grade": grade,
                    "grade_color": grade_color,
                    "description": description,
                    "suggestion": suggestion,
                }

                # 按原料分组存储
                if material not in materials_dict:
                    materials_dict[material] = []
                materials_dict[material].append(img_info)

    # 对每组内的图片按等级排序（优>良>中>差）
    grade_order = {"优": 0, "良": 1, "中": 2, "差": 3}
    for material in materials_dict:
        materials_dict[material].sort(key=lambda x: grade_order.get(x["grade"], 99))

    return templates.TemplateResponse(
        "grading.html",
        {
            "request": request,
            "title": "看图分级教学",
            "materials": materials_dict,  # 传递给模板
        },
    )


@app.get("/recommend", response_class=HTMLResponse)
def recommend_page(request: Request):
    return templates.TemplateResponse(
        "recommend.html",
        {"request": request, "title": "智能推荐"},
    )


@app.get("/api/options")
def get_options():
    return {"question_bank": question_bank}

@app.get("/api/routes")
def list_routes(material: str | None = None):
    """
    轻量知识库接口：按原料类型（可选）返回路线元信息，用于前端展示/搜索。
    """
    df = app.state.routes_df
    if material:
        # 复用推荐器的原料匹配逻辑，而不是简单等值判断
        recommender: RouteRecommender = app.state.recommender
        df = df[df.apply(lambda x: recommender.is_material_match(material, x.get("适用原料类")), axis=1)].copy()
    cols = [c for c in ["路线编号", "路线名称", "适用原料类", "适用等级", "首选目标", "风险等级", "推荐强度", "推荐理由关键词"] if c in df.columns]
    return {"routes": df[cols].fillna("").to_dict(orient="records")}


@app.post("/api/recommend")
def recommend_api(payload: RecommendRequest):
    recommender: RouteRecommender = app.state.recommender
    profile = recommender.normalize_user_profile(payload.user_profile)
    top3 = recommender.recommend_top3(profile)
    if top3 is None:
        raise HTTPException(status_code=400, detail="无可用合规路线，请检查原料情况")

    # 附加解释字段：对Top3逐条补解释
    results = []
    for _, row in top3.iterrows():
        # 从原始routes里定位该路线行，以便解释时拿到适用等级等信息
        route_row = app.state.routes_df[app.state.routes_df["路线编号"] == row["路线编号"]]
        if len(route_row) == 0:
            route_obj = row
        else:
            route_obj = route_row.iloc[0]

        explanation = recommender.explain_route(route_obj, profile)
        results.append({**row.to_dict(), "explanation": explanation})

    return {"user_profile": profile, "recommendations": results}

# ---------- 分级教学辅助函数 ----------
def get_grade_description(material: str, grade: str) -> str:
    """根据原料和等级返回一分钟判别要点"""
    descriptions = {
        ("茶渣", "优"): "色泽均匀、干燥松散、无异味，适合高值化提取。",
        ("茶渣", "良"): "轻微结块，色泽稍暗，仍可考虑材料化利用。",
        ("茶渣", "中"): "有明显受潮痕迹，建议优先选择能源化路线。",
        ("茶渣", "差"): "严重霉变、有酸馊味，禁止食品/提取，建议无害化处置。",
        
        ("果渣", "优"): "新鲜、色泽鲜亮、无褐变，适合提取果胶、多酚等。",
        ("果渣", "良"): "轻微褐变，仍可用于饲料或发酵。",
        ("果渣", "中"): "褐变明显，含水率高，建议能源化或堆肥。",
        ("果渣", "差"): "腐败发臭、霉变严重，禁止一切高值化，仅限无害化。",
        
        ("花生壳/板栗壳", "优"): "干燥、无霉变、无泥沙，适合制备生物炭或板材。",
        ("花生壳/板栗壳", "良"): "轻微受潮，干燥后仍可材料化利用。",
        ("花生壳/板栗壳", "中"): "混有少量泥沙或轻度霉斑，建议能源化。",
        ("花生壳/板栗壳", "差"): "严重霉变或混入大量泥沙，仅限无害化处置。",
        
        ("豆制品/粮食发酵副产物", "优"): "新鲜、无异味、含水率适中，适合饲料或提取。",
        ("豆制品/粮食发酵副产物", "良"): "略有酸味，但仍可用于发酵或能源化。",
        ("豆制品/粮食发酵副产物", "中"): "含水率高，易变质，建议快速能源化。",
        ("豆制品/粮食发酵副产物", "差"): "腐败恶臭，禁止资源化，必须无害化。",
        
        ("中药残余废弃物", "优"): "药材类型明确、干燥、无霉变，可考虑活性成分提取。",
        ("中药残余废弃物", "良"): "轻度吸潮，仍可用于材料化或能源化。",
        ("中药残余废弃物", "中"): "成分复杂或轻微霉变，建议谨慎用于材料。",
        ("中药残余废弃物", "差"): "严重霉变或来源不明，禁止提取，仅限无害化。",
    }
    return descriptions.get((material, grade), "请结合图片观察颜色、霉变、杂质等特征判断。")

def get_grade_suggestion(grade: str) -> str:
    """根据等级给出建议方向"""
    suggestions = {
        "优": "✨ 适合高值化提取、食品级应用",
        "良": "✅ 可考虑材料/农用/部分提取路线",
        "中": "⚠️ 优先选择材料/能源/农用路线，谨慎考虑提取",
        "差": "🚫 禁入食品与精细提取，建议走能源化/无害化兜底",
    }
    return suggestions.get(grade, "")


