import tensorflow as tf
import joblib
from pathlib import Path
import logging
from config import MODEL_PATHS

class ModelLoader:
    def __init__(self):
        self.model = None
        self.scalers = None
        self.config = None
        self.is_loaded = False
        
    def load_all(self):
        """Load model, scalers, and config"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(MODEL_PATHS['model'])
            
            # Load scalers
            self.scalers = joblib.load(MODEL_PATHS['scalers'])
            
            # Load config
            self.config = joblib.load(MODEL_PATHS['config'])
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model components: {e}")
            return False
    
    def get_scaler_for_stock(self, stock_symbol):
        """Get appropriate scaler for stock"""
        if not self.is_loaded:
            return None
            
        # Remove .JK suffix and ^ prefix for symbol matching
        clean_symbol = stock_symbol.replace('.JK', '').replace('^', '')
        
        if clean_symbol in self.scalers:
            return self.scalers[clean_symbol]
        
        # Fallback to first available scaler
        return list(self.scalers.values())[0]
