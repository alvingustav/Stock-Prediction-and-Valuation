import numpy as np
import pandas as pd
from .model_loader import ModelLoader
from .data_collector import DataCollector
from .utils import add_technical_indicators
from config import APP_SETTINGS

class StockPredictor:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.data_collector = DataCollector()
        self.is_ready = False
        
    def initialize(self):
        """Initialize predictor"""
        if self.model_loader.load_all():
            self.is_ready = True
            return True
        return False
    
    def predict_prices(self, stock_symbol, days_ahead=7):
        """Predict stock prices for specified days"""
        if not self.is_ready:
            raise ValueError("Predictor not initialized")
        
        try:
            # Get historical data
            data = self.data_collector.get_stock_data(stock_symbol)
            if data is None:
                return None
            
            # Add technical indicators
            data_with_indicators = add_technical_indicators(data)
            
            # Prepare sequence
            sequence = self._prepare_sequence(data_with_indicators, stock_symbol)
            
            # Generate predictions
            predictions = self._generate_predictions(sequence, days_ahead, stock_symbol)
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error for {stock_symbol}: {e}")
            return None
    
    def _prepare_sequence(self, data, stock_symbol):
        """Prepare data sequence for prediction"""
        scaler_info = self.model_loader.get_scaler_for_stock(stock_symbol)
        feature_columns = self.model_loader.config['feature_columns']
        sequence_length = APP_SETTINGS['sequence_length']
        
        # Get features
        features = data[feature_columns].fillna(method='bfill').fillna(method='ffill')
        
        # Scale features
        scaled_features = scaler_info['feature_scaler'].transform(features.values)
        
        # Create sequence
        if len(scaled_features) >= sequence_length:
            return scaled_features[-sequence_length:]
        else:
            # Pad with last available data if not enough history
            padded = np.tile(scaled_features[-1], (sequence_length, 1))
            padded[-len(scaled_features):] = scaled_features
            return padded
    
    def _generate_predictions(self, sequence, days_ahead, stock_symbol):
        """Generate future predictions"""
        scaler_info = self.model_loader.get_scaler_for_stock(stock_symbol)
        target_scaler = scaler_info['target_scaler']
        
        predictions = []
        current_sequence = sequence.copy()
        
        for _ in range(days_ahead):
            # Predict next value
            pred_input = current_sequence.reshape(1, *current_sequence.shape)
            pred_scaled = self.model_loader.model.predict(pred_input, verbose=0)
            
            # Inverse transform to get actual price
            pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            
            # Update sequence (simplified approach)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # You might want to update with new technical indicators here
        
        return predictions
