# day4_ma_strategy_risklog.py
# Day4: 加入仓位控制、资金检查、交易日志（trades.csv）

import os
import json
import time
import uuid
import logging
import sys
import csv            # === [新增 Day4] 写交易日志用
import datetime as dt
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# =============== 配置 ===============
load_dotenv()

# On 2025-10-11, CRM had a golden cross, good for demo
# RISK_PCT is 0.001, while total equity is 1000k, so max_notional = 1k
SYMBOL = os.getenv("SYMBOL", "CRM")
FEED = os.getenv("DATA_FEED", "iex")
N_TRADING_DAYS = int(os.getenv("N_TRADING_DAYS", "80"))

# 风控参数
ORDER_QTY = float(os.getenv("ORDER_QTY", "1"))
MAX_SHARES = float(os.getenv("MAX_SHARES", "10"))
USE_BRACKET = os.getenv("USE_BRACKET", "true").lower() == "true"
TP_PCT = float(os.getenv("TP_PCT", "0.02"))
SL_PCT = float(os.getenv("SL_PCT", "0.02"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.02"))  # === [新增 Day4] 每次风险不超净值 2%

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
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("day4_risklog.log")]
)

# === [新增 Day4] 交易记录文件 ===
TRADES_CSV = Path("trades.csv")


# =============== 数据函数 ===============
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


# =============== 工具函数 ===============
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
    row = {"ts": time.strftime("%Y-%m-%d"), **entry}
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
                continue
    return out


def already_fired_today(coid: str) -> bool:   # === [修改 Day4] 修正逻辑
    for rec in load_all_states():
        if rec.get("coid") == coid and rec.get("ts") == time.strftime("%Y-%m-%d") and not rec.get("dryrun", False):
            return True
    return False


def log_trade(symbol, side, qty, price, dryrun, result):  # === [新增 Day4]
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), symbol, side, qty, price, dryrun, result])


def print_account_status(api: REST):
    account = api.get_account()
    print("=== Account Status ===")
    print(f"ID: {account.id}")
    print(f"Status: {account.status}")
    print(f"Cash: {account.cash}")
    print(f"Portfolio Value: {account.portfolio_value}")
    print(f"Equity: {account.equity}")
    print("======================")


def print_position(api, symbol):  # === [新增 Day4]
    try:
        pos = api.get_position(symbol)
        logging.info(f"[POS] {symbol}: qty={pos.qty}, avg_entry_price={pos.avg_entry_price}")
    except APIError:
        logging.info(f"[POS] {symbol}: no open position")


# =============== 主流程 ===============
def main():
    api = REST()
    print_account_status(api)

    try:
        clock = api.get_clock()
        if SKIP_IF_CLOSED and not getattr(clock, "is_open", False) and not EXTENDED_HOURS:
            logging.info("Market closed. Skip trading.")
            return
    except APIError as e:
        logging.warning(f"Clock check failed: {e}")

    try:
        df = fetch_bars_df(api, SYMBOL, N_TRADING_DAYS, feed=FEED)
    except APIError as e:
        logging.error(f"Fetch bars failed: {e}")
        return

    if df.empty or "close" not in df.columns:
        logging.info("No bars fetched. Bail out.")
        return

    df = compute_sma(df)
    valid = df.dropna(subset=["SMA5", "SMA20"])
    if len(valid) < 2:
        logging.info("Not enough samples for SMA.")
        return

    prev = valid.iloc[-2]
    last = valid.iloc[-1]
    signal = detect_cross(prev, last)
    logging.info(f"{SYMBOL} last two rows:\n{valid.tail(2)[['close','SMA5','SMA20']]}")

    if not signal:
        logging.info("No crossover signal.")
        return

    # if has_open_orders(api, SYMBOL):
    #     logging.info("Open order exists. Skip.")
    #     return

    # === [新增 Day4] 获取账户资金信息 ===
    acct = api.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)

    pos_qty = get_long_qty(api, SYMBOL)
    coid = new_coid(f"sma-{signal}", SYMBOL)

    if already_fired_today(coid):
        logging.info(f"Skip: same-day same coid => {coid}")
        return

    try:
        price = float(last["close"])

        # === [新增 Day4] 动态仓位计算 ===
        max_notional = equity * RISK_PCT
        qty = max(1, int(max_notional // price))
        if qty > MAX_SHARES:
            qty = MAX_SHARES
        if cash < qty * price:
            logging.info("Insufficient cash, skip order.")
            return

        if signal == "buy":
            if pos_qty > 0:
                logging.info("Already long. Skip buy.")
                return
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
                result = "submitted"
                logging.info(f"Submitted BUY order id={order.id}")
            else:
                logging.info("DRY_RUN=True, skip submit.")
                result = "dry-run"

            append_state({
                "symbol": SYMBOL, "side": "buy", "coid": coid,
                "qty": qty, "price": price, "dryrun": DRY_RUN,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log_trade(SYMBOL, "buy", qty, price, DRY_RUN, result)

        elif signal == "sell":
            if pos_qty <= 0:
                logging.info("No long position. Skip sell.")
                return
            qty = pos_qty
            logging.info(f"[SELL] {SYMBOL} qty={qty} coid={coid}")

            if not DRY_RUN:
                order = api.submit_order(
                    symbol=SYMBOL, qty=qty, side="sell",
                    type="market", time_in_force=TIME_IN_FORCE,
                    extended_hours=EXTENDED_HOURS,
                    client_order_id=coid
                )
                result = "submitted"
                logging.info(f"Submitted SELL order id={order.id}")
            else:
                logging.info("DRY_RUN=True, skip submit.")
                result = "dry-run"

            append_state({
                "symbol": SYMBOL, "side": "sell", "coid": coid,
                "qty": qty, "price": price, "dryrun": DRY_RUN,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log_trade(SYMBOL, "sell", qty, price, DRY_RUN, result)

        print_position(api, SYMBOL)  # === [新增 Day4] 调试辅助

    except APIError as e:
        logging.error(f"Order failed: {e}")


if __name__ == "__main__":
    main()
