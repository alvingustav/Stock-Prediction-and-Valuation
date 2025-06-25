import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """Add technical indicators to stock data"""
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price ratios
    df['Price_change'] = df['Close'].pct_change()
    df['High_Low_ratio'] = df['High'] / df['Low']
    df['Open_Close_ratio'] = df['Open'] / df['Close']
    
    return df

def calculate_price_metrics(data):
    """Calculate price performance metrics"""
    current_price = data['Close'].iloc[-1]
    
    # Price changes
    daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    daily_change_pct = (daily_change / data['Close'].iloc[-2]) * 100
    
    # 52-week high/low
    high_52w = data['High'].max()
    low_52w = data['Low'].min()
    
    # Volatility
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    
    return {
        'current_price': current_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'volatility': volatility
    }
