import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            if symbol in self.cache:
                return self.cache[symbol]
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if not data.empty:
                self.cache[symbol] = data
                return data
            return None
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def get_latest_price(self, symbol):
        """Get latest price for a stock"""
        data = self.get_stock_data(symbol, period='1d')
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
