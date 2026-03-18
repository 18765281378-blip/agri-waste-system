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
    # 自动扫描 web/static/grading 下的图片文件
    grading_dir = STATIC_DIR / "grading"
    grading_images = []
    if grading_dir.exists() and grading_dir.is_dir():
        for fp in sorted(grading_dir.iterdir()):
            if fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                grading_images.append(
                    {
                        "filename": fp.name,
                        # 访问路径：/static/grading/文件名
                        "url": f"/static/grading/{fp.name}",
                    }
                )

    return templates.TemplateResponse(
        "grading.html",
        {
            "request": request,
            "title": "看图分级教学",
            "grading_images": grading_images,
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

