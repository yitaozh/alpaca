# day5_backtest_ma.py
# 回测：SMA5/20 均线交叉（次日开盘成交），含交易日志与绩效指标
# 依赖: yfinance, pandas, numpy, matplotlib

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------- 可调参数 -------------
symbols = ["NVDA"]#, "META", "AMZN", "SMCI", "PLTR", "TSLA"] # 多票回测
START = "2024-01-01"           # 起始日期
END = None                     # 截止日期，None=今天
FAST = 5                       # 快均线
SLOW = 20                      # 慢均线
INIT_CAPITAL = 100000          # 初始资金
SLIPPAGE_BPS = 1.0             # 滑点(bps): 1 = 0.01%
FEE_PER_TRADE = 0.0            # 每笔固定费用（美元）
ALLOW_FRACTIONAL = False       # 是否允许小数股
RISK_ALLOCATION = 1.0          # 每次买入使用的资金比例(0~1)，1=满仓
PLOT = True                    # 是否画图
# ------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_px: float
    shares: float
    exit_date: pd.Timestamp = None
    exit_px: float = None

def fetch_ohlc(symbol: str, start: str, end: str = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("数据为空，检查代码或时间范围")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df

def build_signals(df: pd.DataFrame, fast=FAST, slow=SLOW) -> pd.DataFrame:
    out = df.copy()
    out["SMA_F"] = out["Close"].rolling(fast, min_periods=fast).mean()
    out["SMA_S"] = out["Close"].rolling(slow, min_periods=slow).mean()
    # 金叉: 昨天 SMAF<=SMAS, 今天 SMAF>SMAS
    cond_golden = (out["SMA_F"].shift(1) <= out["SMA_S"].shift(1)) & (out["SMA_F"] > out["SMA_S"])
    # 死叉: 昨天 SMAF>=SMAS, 今天 SMAF<SMAS
    cond_dead = (out["SMA_F"].shift(1) >= out["SMA_S"].shift(1)) & (out["SMA_F"] < out["SMA_S"])
    out["signal"] = np.where(cond_golden, "BUY", np.where(cond_dead, "SELL", "HOLD"))
    return out

def apply_slip(price: float, is_buy: bool, slippage_bps: float) -> float:
    """按bps加滑点：买贵卖便宜"""
    mult = 1.0 + (slippage_bps / 10000.0) * (1 if is_buy else -1)
    return price * mult

def shares_to_buy(cash: float, px: float, risk_alloc: float, allow_fractional: bool) -> float:
    notional = cash * max(0.0, min(1.0, risk_alloc))
    if allow_fractional:
        return notional / px if px > 0 else 0.0
    q = int(notional // px) if px > 0 else 0
    return float(q)

def backtest_long_only(df_sig: pd.DataFrame,
                       init_capital=INIT_CAPITAL,
                       slippage_bps=SLIPPAGE_BPS,
                       fee=FEE_PER_TRADE,
                       allow_fractional=ALLOW_FRACTIONAL,
                       risk_alloc=RISK_ALLOCATION):
    cash = init_capital
    position = 0.0
    position_cost = 0.0  # 持仓均价可选，这里只做持仓市值计算
    trades: list[Trade] = []

    # 资金曲线：按收盘计算
    equity_curve = []

    dates = df_sig.index

    for i in range(1, len(dates)):  # 从第二天开始，因为交易按“次日开盘”
        today = dates[i]
        yesterday = dates[i - 1]

        # 先根据昨日信号，在今日开盘执行
        sigs = df_sig.loc[yesterday, "signal"]
        if isinstance(sigs, pd.Series):
            sig = sigs.iloc[0]
        else:
            sig = sigs

        today_open = float(df_sig.loc[today, "Open"])
        if math.isnan(today_open):
            # 若缺开盘价，跳过执行
            pass
        else:
            if sig == "BUY" and position <= 1e-12:
                px = apply_slip(today_open, is_buy=True, slippage_bps=slippage_bps)
                qty = shares_to_buy(cash, px, risk_alloc, allow_fractional)
                if qty >= (0.0001 if allow_fractional else 1):
                    cost = qty * px + fee
                    if cost <= cash + 1e-9:
                        cash -= cost
                        position += qty
                        position_cost = px
                        trades.append(Trade(entry_date=today, entry_px=px, shares=qty))
            elif sig == "SELL" and position > 1e-12:
                px = apply_slip(today_open, is_buy=False, slippage_bps=slippage_bps)
                proceeds = position * px - fee
                cash += proceeds
                # 完成一笔交易
                if trades and trades[-1].exit_date is None:
                    trades[-1].exit_date = today
                    trades[-1].exit_px = px
                position = 0.0
                position_cost = 0.0

        # 记录资金曲线（用今日收盘估值）
        close_px = float(df_sig.loc[today, "Close"])
        mkt_value = position * close_px
        equity = cash + mkt_value
        equity_curve.append((today, equity))

    # 若最后还持仓，按最后一天收盘价平仓用于绩效统计（不影响现金，只为闭合交易）
    if position > 1e-12:
        last_day = dates[-1]
        last_close = float(df_sig.loc[last_day, "Close"])
        if trades and trades[-1].exit_date is None:
            trades[-1].exit_date = last_day
            trades[-1].exit_px = last_close

    curve = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return curve, trades

def perf_metrics(curve: pd.DataFrame, risk_free_rate_annual=0.0):
    if curve.empty:
        return {}

    # 日收益
    ret = curve["equity"].pct_change().dropna()
    if ret.empty:
        return {}

    # CAGR
    total_return = curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0
    days = (curve.index[-1] - curve.index[0]).days
    years = max(1e-9, days / 365.25)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -0.999999 else -1.0

    # 年化波动、Sharpe
    vol_annual = ret.std() * np.sqrt(252)
    mean_annual = ret.mean() * 252
    sharpe = (mean_annual - risk_free_rate_annual) / vol_annual if vol_annual > 1e-12 else np.nan

    # 最大回撤
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    max_dd = dd.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility(ann.)": vol_annual,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Start": curve.index[0].date(),
        "End": curve.index[-1].date(),
    }

def trades_summary(trades: list[Trade]):
    if not trades:
        return {"Trades": 0, "Win Rate": np.nan, "Avg Return": np.nan}

    # 只统计已平仓
    closed = [t for t in trades if t.exit_date is not None]
    if not closed:
        return {"Trades": 0, "Win Rate": np.nan, "Avg Return": np.nan}

    rets = [(t.exit_px - t.entry_px) / t.entry_px for t in closed]
    wins = [r for r in rets if r > 0]
    return {
        "Trades": len(closed),
        "Win Rate": len(wins) / len(closed) if closed else np.nan,
        "Avg Return": float(np.mean(rets)) if rets else np.nan
    }

def plot_equity(curve: pd.DataFrame, symbol: str):
    plt.figure(figsize=(10, 4))
    curve["equity"].plot()
    plt.title(f"Equity Curve - {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_backtest_for_symbol(SYMBOL: str):
    print(f"[Backtest] {SYMBOL} {START} ~ {END or 'today'} | SMA{FAST}/{SLOW}")
    df = fetch_ohlc(SYMBOL, START, END)
    sig = build_signals(df, FAST, SLOW)

    curve, trades = backtest_long_only(
        sig,
        init_capital=INIT_CAPITAL,
        slippage_bps=SLIPPAGE_BPS,
        fee=FEE_PER_TRADE,
        allow_fractional=ALLOW_FRACTIONAL,
        risk_alloc=RISK_ALLOCATION
    )
    perf = perf_metrics(curve)
    tsum = trades_summary(trades)

    print("\n=== Performance ===")
    for k, v in perf.items():
        if isinstance(v, float):
            if "Drawdown" in k:
                print(f"{k:>16}: {v:.2%}")
            elif k in ("Volatility(ann.)", "Sharpe"):
                print(f"{k:>16}: {v:.4f}")
            else:
                print(f"{k:>16}: {v:.2%}")
        else:
            print(f"{k:>16}: {v}")

    print("\n=== Trades (closed) ===")
    print(f"Trades: {tsum['Trades']}, Win Rate: {tsum['Win Rate'] if not pd.isna(tsum['Win Rate']) else np.nan:.2%}" if tsum['Trades'] else "Trades: 0")
    if tsum['Trades']:
        print(f"Avg Return per trade: {tsum['Avg Return']:.2%}")

    # === 收益评估 ===
    initial_price = float(df['Close'].iloc[0])
    final_price = float(df['Close'].iloc[-1])
    hold_return = (final_price / initial_price - 1) * 100

    initial_equity = float(curve.iloc[0])
    final_equity = float(curve.iloc[-1])
    strat_return = (final_equity / initial_equity - 1) * 100

    excess = strat_return - hold_return

    print("\n=== 📈 策略绩效报告 ===")
    print(f"策略收益: {strat_return:+.2f}%")
    print(f"Buy & Hold 收益: {hold_return:+.2f}%")
    print(f"超额收益: {excess:+.2f}%")
    print("========================\n")

    # 导出交易日志
    if trades:
        rows = []
        for t in trades:
            rows.append({
                "entry_date": t.entry_date,
                "entry_px": t.entry_px,
                "exit_date": t.exit_date,
                "exit_px": t.exit_px,
                "shares": t.shares,
                "ret_pct": None if t.exit_px is None else (t.exit_px - t.entry_px) / t.entry_px
            })
        pd.DataFrame(rows).to_csv("day5_trades.csv", index=False)
        print("\nSaved trades to day5_trades.csv")

    # 画图
    if PLOT:
        plot_equity(curve, SYMBOL)

def main():
    for symbol in symbols:
        run_backtest_for_symbol(symbol)

if __name__ == "__main__":
    main()
