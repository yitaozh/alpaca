# day3_ma_strategy_simple.py
# Day3: 基于极简 fetch 逻辑的均线交叉策略（SMA5/20），带风控与幂等

import os
import json
import time
import uuid
import logging
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# =============== 配置 ===============
load_dotenv()  # 可用 .env 覆盖

SYMBOL = os.getenv("SYMBOL", "MMM")
FEED = os.getenv("DATA_FEED", "iex")         # 免费账号用 iex
N_TRADING_DAYS = int(os.getenv("N_TRADING_DAYS", "80"))
# 用自然日 80，能覆盖 ~50 个交易日，别抬杠；不够自己改大点

# 风控
ORDER_QTY = float(os.getenv("ORDER_QTY", "1"))
MAX_SHARES = float(os.getenv("MAX_SHARES", "10"))
USE_BRACKET = os.getenv("USE_BRACKET", "true").lower() == "true"
TP_PCT = float(os.getenv("TP_PCT", "0.02"))  # 止盈 2%
SL_PCT = float(os.getenv("SL_PCT", "0.02"))  # 止损 2%

# 行为
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TIME_IN_FORCE = os.getenv("TIME_IN_FORCE", "day")
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() == "true"
SKIP_IF_CLOSED = os.getenv("SKIP_IF_CLOSED", "false").lower() == "true"

# 状态与日志
STATE_PATH = Path(".state/state.log")
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("day3_simple.log")]
)

# =============== 极简数据函数（按你要求） ===============
def fetch_bars_df(api: REST, symbol: str, n_trading_days: int, feed: str = "iex"):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=n_trading_days)
    df = api.get_bars(
        symbol,
        TimeFrame.Day,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        adjustment="raw",
        feed=feed
    ).df
    # Alpaca 可能返回 MultiIndex（多标的样式），单票时取出该票
    if isinstance(df.index, pd.MultiIndex) and symbol in df.index.levels[0]:
        df = df.loc[symbol]
    return df

def compute_sma(df: pd.DataFrame):
    out = df.copy()
    out["SMA5"] = out["close"].rolling(5, min_periods=5).mean()
    out["SMA20"] = out["close"].rolling(20, min_periods=20).mean()
    return out

def detect_cross(prev_row, last_row):
    if prev_row["SMA5"] <= prev_row["SMA20"] and last_row["SMA5"] > last_row["SMA20"]:
        return "buy"
    if prev_row["SMA5"] >= prev_row["SMA20"] and last_row["SMA5"] < last_row["SMA20"]:
        return "sell"
    return ""

# =============== 小工具 ===============
def has_open_orders(api: REST, symbol: str) -> bool:
    return len(api.list_orders(status="open", symbols=[symbol], limit=50)) > 0

def get_long_qty(api: REST, symbol: str) -> float:
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except APIError:
        return 0.0

def new_coid(prefix: str, symbol: str) -> str:
    return f"{prefix}-{symbol}-{uuid.uuid4().hex[:8]}"

def append_state(entry: dict):
    # 自动加上时间戳
    row = {
        "ts": time.strftime("%Y-%m-%d"),
        **entry,
    }
    with open(STATE_PATH, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_all_states():
    if not os.path.exists(STATE_PATH):
        return []
    out = []
    with open(STATE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # 坏行就跳过，别让策略崩
                continue
    return out

def already_fired_today() -> bool:
    """
    如果是 dryrun，则忽略之前 dryrun 的标记。
    """
    for rec in load_all_states():
        if (rec.get("ts") == time.strftime("%Y-%m-%d")) and rec.get("dryrun") == False:
            return True

def print_account_status(api: REST):
    account = api.get_account()
    print("=== Account Status ===")
    print(f"ID: {account.id}")
    print(f"Status: {account.status}")  # ACTIVE, REJECTED, etc.
    print(f"Cash: {account.cash}")
    print(f"Portfolio Value: {account.portfolio_value}")
    print(f"Trading Blocked: {account.trading_blocked}")
    print(f"Account Blocked: {account.account_blocked}")
    print(f"Pattern Day Trader: {account.pattern_day_trader}")
    print(f"Shorting Enabled: {account.shorting_enabled}")
    print(f"Equity: {account.equity}")
    print("======================")

# =============== 主流程 ===============
def main():
    api = REST()  # 从环境变量读取 APCA_API_KEY_ID / APCA_API_SECRET_KEY / APCA_API_BASE_URL
    print_account_status(api)
    # 时段检查
    try:
        clock = api.get_clock()
        if SKIP_IF_CLOSED and not getattr(clock, "is_open", False) and not EXTENDED_HOURS:
            logging.info("Market closed. Skip trading.")
            return
    except APIError as e:
        logging.warning(f"Clock check failed: {e}")

    # 拉数据（极简逻辑）
    try:
        df = fetch_bars_df(api, SYMBOL, N_TRADING_DAYS, feed=FEED)
    except APIError as e:
        logging.error(f"Fetch bars failed: {e}")
        return

    if df.empty or "close" not in df.columns:
        logging.info("No bars fetched. Bail out.")
        return

    # 计算均线与信号
    df = compute_sma(df)
    valid = df.dropna(subset=["SMA5", "SMA20"])
    if len(valid) < 2:
        logging.info("Not enough samples for SMA.")
        return

    prev = valid.iloc[-2]
    last = valid.iloc[-1]
    signal = detect_cross(prev, last)
    last_date = last.name.date().isoformat() if hasattr(last.name, "date") else str(last.name)

    logging.info(f"{SYMBOL} last two rows:\n{valid.tail(2)[['close','SMA5','SMA20']]}")

    if not signal:
        logging.info("No crossover signal.")
        return

    # 未完成订单检查
    if has_open_orders(api, SYMBOL):
        logging.info("Open order exists. Skip.")
        return

    # 仓位与下单
    pos_qty = get_long_qty(api, SYMBOL)
    coid = new_coid(f"sma-{signal}", SYMBOL)

    # 防重复：同一天同 coid 直接跳过
    if already_fired_today():
        logging.info(f"Skip: same-day same coid seen => {coid}")
        return

    try:
        if signal == "buy":
            if pos_qty > 0:
                logging.info("Already long. Skip buy.")
                return
            qty = min(ORDER_QTY, MAX_SHARES)
            price = float(last["close"])
            tp = round(price * (1 + TP_PCT), 2)
            sl = round(price * (1 - SL_PCT), 2)
            logging.info(f"[BUY] {SYMBOL} qty={qty} price≈{price} tp={tp} sl={sl} coid={coid}")

            if not DRY_RUN:
                if USE_BRACKET:
                    order = api.submit_order(
                        symbol=SYMBOL, qty=qty, side="buy",
                        type="market", time_in_force=TIME_IN_FORCE,
                        order_class="bracket",
                        take_profit={"limit_price": tp},
                        stop_loss={"stop_price": sl},
                        extended_hours=EXTENDED_HOURS,
                        client_order_id=coid
                    )
                else:
                    order = api.submit_order(
                        symbol=SYMBOL, qty=qty, side="buy",
                        type="market", time_in_force=TIME_IN_FORCE,
                        extended_hours=EXTENDED_HOURS,
                        client_order_id=coid
                    )
                logging.info(f"Submitted BUY order id={order.id} coid={order.client_order_id}")
            else:
                logging.info("DRY_RUN=True, not submitting buy order.")

            append_state({
                "symbol": SYMBOL,
                "side": "buy",
                "coid": coid,
                "qty": qty,
                "price": float(last["close"]),
                "dryrun": DRY_RUN,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        elif signal == "sell":
            if pos_qty <= 0:
                logging.info("No long position. Skip sell.")
                return
            qty = pos_qty  # 全平；想减仓就改成 min(ORDER_QTY, pos_qty)
            logging.info(f"[SELL] {SYMBOL} qty={qty} coid={coid}")

            if not DRY_RUN:
                order = api.submit_order(
                    symbol=SYMBOL, qty=qty, side="sell",
                    type="market", time_in_force=TIME_IN_FORCE,
                    extended_hours=EXTENDED_HOURS,
                    client_order_id=coid
                )
                logging.info(f"Submitted SELL order id={order.id} coid={order.client_order_id}")
            else:
                logging.info("DRY_RUN=True, not submitting sell order.")

            append_state({
                "symbol": SYMBOL,
                "side": "sell",
                "coid": coid,
                "qty": qty,
                "price": float(last["close"]),
                "dryrun": DRY_RUN,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    except APIError as e:
        logging.error(f"Order failed: {e}")

if __name__ == "__main__":
    main()
