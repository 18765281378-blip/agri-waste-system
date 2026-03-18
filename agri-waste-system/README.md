## 项目简介

本项目实现“工艺知识库 + 数据驱动推荐”的副产物处置方案智能网站雏形：

- **看图分级教学**：`/grading`
- **智能推荐**：`/recommend`（红线规则过滤 → 模型打分排序 Top3 → 可解释输出）
- **API**：
  - `GET /api/routes?material=...`：按原料筛路线元信息
  - `POST /api/recommend`：输入用户画像，返回 Top3 方案卡 + 解释

## 快速启动（推荐使用虚拟环境）

在项目根目录（有 `web_app.py` 的目录）执行。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=300 --retries=10 -r requirements.txt
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

浏览器打开：

- 首页：`http://127.0.0.1:8000/`
- 看图分级：`http://127.0.0.1:8000/grading`
- 智能推荐：`http://127.0.0.1:8000/recommend`

## 说明

- **等级“中”**：当前模型训练数据只包含“优/良/差”，网站允许选择“中”，会临时按“良”进入模型以保证兼容；后续你补齐“优/良/中/差”训练数据后可再训练升级。
- **可解释输出**：推荐结果里包含 `core_reasons`（原料/等级匹配）与 `feature_contributions_top`（若模型支持 `pred_contrib`）。

