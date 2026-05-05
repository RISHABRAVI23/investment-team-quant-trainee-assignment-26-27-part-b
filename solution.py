import pandas as pd
import numpy as np
import vectorbt as vbt

df = pd.read_csv('/content/drive/MyDrive/Quant Investment team trainee assignment/Data.csv')


stocks = ['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E']
columns = ['timestamp']


for stock in stocks:
    columns += [f'{stock}_Close', f'{stock}_Open', f'{stock}_High', f'{stock}_Low', f'{stock}_Volume']

df.columns = columns
df.set_index('timestamp', inplace=True)

df.replace(0, np.nan, inplace=True)
df.ffill(inplace=True) #forward fill

def get_indicators(close_prices, rsi_window=14, vol_window=20):
    # RSI calculation using vectorbt for efficiency
    rsi = vbt.RSI.run(close_prices, window=rsi_window).rsi
    
    # Volatility: Standard Deviation of price changes
    returns = close_prices.pct_change()
    volatility = returns.rolling(window=vol_window).std()
    
    # Moving Average of Volatility for the filter
    vol_ma = volatility.rolling(window=vol_window).mean()
    
    return rsi, volatility, vol_ma