import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading环境地址

# 创建API连接
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# 查看账户信息
account = api.get_account()
print("账户状态:", account.status)
print("初始余额:", account.cash)

# 下一个小额买单：买入1股AAPL
order = api.submit_order(
    symbol='AAPL',
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc'
)

print("下单成功:", order)
