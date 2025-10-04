# %% 导入库 + 设置 API
import os
import sys
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
from dotenv import load_dotenv
import datetime as dt

# 加载环境变量
load_dotenv()

# 建立连接
api = REST()

# %% 获取数据
# 取历史数据：过去 50 天，日线
symbol = "AAPL"
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=80)
bars = api.get_bars(
    symbol,
    TimeFrame.Day,
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
    adjustment="raw",
    feed="iex"
).df
bars = bars.tail(50)

# %% Debug
print(bars)

# %% 计算均线
# 计算均线
bars["SMA5"] = bars["close"].rolling(window=5).mean()
bars["SMA20"] = bars["close"].rolling(window=20).mean()

print(bars.head())

# %% 样本检查
valid = bars.dropna(subset=["SMA5", "SMA20"])
if len(valid) < 2:
    print("样本不足，跳过信号判断")
    sys.exit(1)

# %% 最近两天数据
last = bars.iloc[-1]
prev = bars.iloc[-2]

print(bars.tail(5))  # 打印看看最后几行数据

# %% 策略信号
if prev["SMA5"] <= prev["SMA20"] and last["SMA5"] > last["SMA20"]:
    print("产生买入信号: AAPL")
    api.submit_order(
        symbol=symbol,
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc"
    )

elif prev["SMA5"] >= prev["SMA20"] and last["SMA5"] < last["SMA20"]:
    print("产生卖出信号: AAPL")
    api.submit_order(
        symbol=symbol,
        qty=1,
        side="sell",
        type="market",
        time_in_force="gtc"
    )
else:
    print("没有信号，保持持仓不变")

# %%
