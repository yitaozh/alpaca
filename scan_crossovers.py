# -*- coding: utf-8 -*-
"""
批量扫描最近两天是否发生均线交叉（SMA5 vs SMA20）
默认扫描 50 个美股热门标的（科技/大盘/金融/消费）
"""

import datetime as dt
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame, APIError
from dotenv import load_dotenv

# === 默认 50 个标的 ===
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "IBM", "ORCL", "CRM", "ADBE", "PYPL", "QCOM", "CSCO", "AVGO", "TXN", "AMAT",
    "BA", "CAT", "GE", "MMM", "HON", "UNP", "UPS", "FDX", "NKE", "SBUX",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "AXP", "BLK",
    "WMT", "COST", "HD", "LOW", "TGT", "MCD", "KO", "PEP", "DIS", "PG"
]

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
    return df

def compute_sma(df: pd.DataFrame):
    out = df.copy()
    out["SMA5"] = out["close"].rolling(5, min_periods=5).mean()
    out["SMA20"] = out["close"].rolling(20, min_periods=20).mean()
    return out

def detect_cross(prev_row, last_row):
    if prev_row["SMA5"] <= prev_row["SMA20"] and last_row["SMA5"] > last_row["SMA20"]:
        return "golden"
    if prev_row["SMA5"] >= prev_row["SMA20"] and last_row["SMA5"] < last_row["SMA20"]:
        return "dead"
    return ""

def scan_symbols(symbols, n_trading_days: int = 50, feed: str = "iex"):
    api = REST()
    results = []

    for sym in symbols:
        try:
            df = fetch_bars_df(api, sym, n_trading_days, feed=feed)
            if df.empty or "close" not in df.columns:
                continue
            df = compute_sma(df)
            valid = df.dropna(subset=["SMA5", "SMA20"])
            if len(valid) < 2:
                continue
            prev, last = valid.iloc[-2], valid.iloc[-1]
            tag = detect_cross(prev, last)
            if tag:
                results.append({
                    "symbol": sym,
                    "prev_date": prev.name.date(),
                    "last_date": last.name.date(),
                    "prev_SMA5": round(prev["SMA5"], 2),
                    "prev_SMA20": round(prev["SMA20"], 2),
                    "last_SMA5": round(last["SMA5"], 2),
                    "last_SMA20": round(last["SMA20"], 2),
                    "cross": tag
                })
        except APIError as e:
            print(f"[WARN] {sym} 拉取失败: {e}")
        except Exception as e:
            print(f"[WARN] {sym} 异常: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    load_dotenv()
    df = scan_symbols(DEFAULT_SYMBOLS, n_trading_days=50, feed="iex")
    if df.empty:
        print("50 个标的里没有找到最近两天发生交叉的。")
    else:
        print(df.to_string(index=False))
