import sqlite3
import datetime
import random
from contextlib import contextmanager
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from collections import deque

# 创建路由
router = APIRouter(prefix="/api/carbon-market", tags=["碳积分市场"])

# ---------- 数据库初始化 ----------
DB_PATH = "carbon_market.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                balance REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 碳积分记录表（来源：碳足迹计算）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS carbon_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                source TEXT,           -- "carbon_footprint"
                reduction_kg REAL,     -- 实际减排量 (kg CO₂e)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        # 订单表（挂单）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                type TEXT,             -- "buy" 或 "sell"
                price REAL,            -- 价格 (积分/吨 CO₂e)
                quantity REAL,         -- 数量 (吨 CO₂e)
                remaining REAL,        -- 剩余未成交数量
                status TEXT DEFAULT 'active',  -- active, completed, cancelled
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        # 成交记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                buy_order_id INTEGER,
                sell_order_id INTEGER,
                price REAL,
                quantity REAL,
                traded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 模拟K线数据表（存储每日收盘价）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        conn.commit()

# 上下文管理器方便操作
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ---------- 辅助函数 ----------
def get_or_create_user(user_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (user_id, balance) VALUES (?, 0)", (user_id,))
            return {"user_id": user_id, "balance": 0}
        return dict(user)

def add_carbon_points(user_id: str, amount: float, source: str, reduction_kg: float):
    """发放碳积分（1积分 = 1 kg CO₂e 减排）"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 更新用户余额
        cursor.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
        # 记录来源
        cursor.execute(
            "INSERT INTO carbon_records (user_id, amount, source, reduction_kg) VALUES (?, ?, ?, ?)",
            (user_id, amount, source, reduction_kg)
        )
        return {"success": True, "new_balance": get_or_create_user(user_id)["balance"]}

# ---------- 订单匹配引擎 ----------
def match_orders():
    """尝试匹配所有活跃的买单和卖单"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 获取所有活跃买单（按价格从高到低）
        cursor.execute("SELECT * FROM orders WHERE type='buy' AND status='active' ORDER BY price DESC, created_at ASC")
        buy_orders = [dict(row) for row in cursor.fetchall()]
        # 获取所有活跃卖单（按价格从低到高）
        cursor.execute("SELECT * FROM orders WHERE type='sell' AND status='active' ORDER BY price ASC, created_at ASC")
        sell_orders = [dict(row) for row in cursor.fetchall()]

        trades = []
        i, j = 0, 0
        while i < len(buy_orders) and j < len(sell_orders):
            buy = buy_orders[i]
            sell = sell_orders[j]
            if buy['price'] >= sell['price']:
                # 可成交
                quantity = min(buy['remaining'], sell['remaining'])
                price = (buy['price'] + sell['price']) / 2  # 取中间价
                # 记录成交
                cursor.execute(
                    "INSERT INTO trades (buy_order_id, sell_order_id, price, quantity) VALUES (?, ?, ?, ?)",
                    (buy['order_id'], sell['order_id'], price, quantity)
                )
                # 更新剩余数量
                buy['remaining'] -= quantity
                sell['remaining'] -= quantity
                cursor.execute("UPDATE orders SET remaining = ? WHERE order_id = ?", (buy['remaining'], buy['order_id']))
                cursor.execute("UPDATE orders SET remaining = ? WHERE order_id = ?", (sell['remaining'], sell['order_id']))
                # 如果订单完成，更新状态
                if buy['remaining'] <= 0:
                    cursor.execute("UPDATE orders SET status = 'completed' WHERE order_id = ?", (buy['order_id'],))
                    i += 1
                if sell['remaining'] <= 0:
                    cursor.execute("UPDATE orders SET status = 'completed' WHERE order_id = ?", (sell['order_id'],))
                    j += 1
                # 转移碳积分（买方余额减少，卖方余额增加）
                # 从买方扣款（扣减余额），买方获得 quantity 吨的碳积分（但余额本就是积分，逻辑要清晰：余额是积分数量）
                # 买方支付 price * quantity 积分给卖方，同时买方获得 quantity 吨的碳权？注意：碳积分本身就是可交易资产。
                # 简化：买卖双方直接转账积分。买方余额减少 price*quantity，卖方余额增加 price*quantity。
                # 注意：这里的余额就是碳积分数量（单位：kg CO₂e）。
                with get_db() as conn2:
                    cur2 = conn2.cursor()
                    # 买方扣款
                    cur2.execute("UPDATE users SET balance = balance - ? WHERE user_id = ?", (price * quantity, buy['user_id']))
                    # 卖方加款
                    cur2.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (price * quantity, sell['user_id']))
                trades.append({"buy_order_id": buy['order_id'], "sell_order_id": sell['order_id'], "price": price, "quantity": quantity})
            else:
                break
        return trades

# ---------- 模拟K线数据生成 ----------
def generate_mock_kline():
    """生成模拟K线数据（最近30天）"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 检查是否已有数据
        cursor.execute("SELECT COUNT(*) FROM market_prices")
        if cursor.fetchone()[0] > 0:
            return
        # 生成30天模拟数据
        base_price = 10.0
        data = []
        for i in range(30):
            date = (datetime.datetime.now() - datetime.timedelta(days=30-i)).strftime("%Y-%m-%d")
            open_price = base_price + random.uniform(-2, 2)
            close_price = open_price + random.uniform(-1.5, 1.5)
            high = max(open_price, close_price) + random.uniform(0, 1)
            low = min(open_price, close_price) - random.uniform(0, 1)
            volume = random.uniform(50, 200)
            cursor.execute(
                "INSERT INTO market_prices (date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)",
                (date, round(open_price,2), round(high,2), round(low,2), round(close_price,2), round(volume,2))
            )
            base_price = close_price

# ---------- API 接口 ----------
class CarbonPointsRequest(BaseModel):
    user_id: str
    reduction_kg: float   # 减排量（kg CO₂e）
    source: str = "carbon_footprint"

class OrderRequest(BaseModel):
    user_id: str
    type: str             # "buy" 或 "sell"
    price: float
    quantity: float

# 初始化数据库和K线数据（在应用启动时调用）
def init_carbon_market():
    init_db()
    generate_mock_kline()

@router.post("/points/add")
def add_points(req: CarbonPointsRequest):
    """发放碳积分（1积分 = 1 kg CO₂e）"""
    amount = req.reduction_kg  # 1:1转换
    result = add_carbon_points(req.user_id, amount, req.source, req.reduction_kg)
    return result

@router.get("/balance/{user_id}")
def get_balance(user_id: str):
    user = get_or_create_user(user_id)
    return {"user_id": user_id, "balance": user["balance"]}

@router.post("/order")
def place_order(req: OrderRequest):
    """挂单"""
    if req.type not in ["buy", "sell"]:
        raise HTTPException(400, "订单类型错误")
    user = get_or_create_user(req.user_id)
    if req.type == "buy":
        # 检查余额是否足够（买单需要冻结资金，这里直接扣减？为简单，挂单时不扣，成交时扣。但为了防止超卖，可以检查可用余额）
        # 这里不预扣，成交时扣减。注意：可能用户下单后余额不足导致成交失败，匹配时会检查。
        pass
    else:
        # 卖单需要检查用户是否有足够的碳积分库存
        if user["balance"] < req.quantity:
            raise HTTPException(400, f"碳积分不足，当前余额：{user['balance']} kg CO₂e")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO orders (user_id, type, price, quantity, remaining) VALUES (?, ?, ?, ?, ?)",
            (req.user_id, req.type, req.price, req.quantity, req.quantity)
        )
        order_id = cursor.lastrowid
    # 尝试匹配
    match_orders()
    return {"order_id": order_id, "status": "active"}

@router.get("/orders/{user_id}")
def get_user_orders(user_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC", (user_id,))
        orders = [dict(row) for row in cursor.fetchall()]
        return {"orders": orders}

@router.get("/market/kline")
def get_kline(days: int = 30):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM market_prices ORDER BY date DESC LIMIT ?", (days,))
        rows = cursor.fetchall()
        # 转为列表，日期升序
        data = [dict(row) for row in reversed(rows)]
        return {"data": data}

@router.get("/market/orderbook")
def get_orderbook():
    """获取市场深度（买一卖一）"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price, SUM(remaining) as total FROM orders WHERE type='buy' AND status='active' GROUP BY price ORDER BY price DESC")
        buys = [{"price": row[0], "quantity": row[1]} for row in cursor.fetchall()]
        cursor.execute("SELECT price, SUM(remaining) as total FROM orders WHERE type='sell' AND status='active' GROUP BY price ORDER BY price ASC")
        sells = [{"price": row[0], "quantity": row[1]} for row in cursor.fetchall()]
        return {"buys": buys, "sells": sells}