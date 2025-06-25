import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

# Now import modules
from data_collector import DataCollector
from model_loader import ModelLoader
from predictor import StockPredictor
from valuation_analyzer import ValuationAnalyzer
from utils import add_technical_indicators, calculate_price_metrics
from config import INDONESIAN_STOCKS, AZURE_OPENAI_CONFIG, APP_SETTINGS

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# App configuration
st.set_page_config(
    page_title="Indonesia Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# Initialize components once
@st.cache_resource
def init_components():
    try:
        data_collector = DataCollector()
        model_loader = ModelLoader()
        predictor = StockPredictor()
        valuation_analyzer = ValuationAnalyzer()
        
        # Initialize predictor
        if predictor.initialize():
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Failed to load model")
            
        return data_collector, predictor, valuation_analyzer
        
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None

# Main app
st.title("🇮🇩 Indonesia Stock Prediction & Valuation")
st.markdown("*Prediksi harga saham menggunakan Deep Learning dan Azure OpenAI*")

# Sidebar
with st.sidebar:
    st.header("📊 Settings")
    
    selected_stock_name = st.selectbox(
        "Pilih Saham:",
        list(INDONESIAN_STOCKS.keys())
    )
    
    selected_stock_code = INDONESIAN_STOCKS[selected_stock_name]
    
    prediction_days = st.slider(
        "Prediksi (hari):",
        min_value=1,
        max_value=APP_SETTINGS['max_prediction_days'],
        value=APP_SETTINGS['default_prediction_days']
    )

# Load components
data_collector, predictor, valuation_analyzer = init_components()

if all([data_collector, predictor, valuation_analyzer]):
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🤖 Price Prediction", "💰 Valuation Report"])
    
    with tab1:
        st.subheader(f"📊 Analysis for {selected_stock_name}")
        
        # Get stock data
        with st.spinner("Fetching stock data..."):
            stock_data = data_collector.get_stock_data(selected_stock_code, period='1y')
        
        if stock_data is not None and not stock_data.empty:
            # Calculate metrics
            metrics = calculate_price_metrics(stock_data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"Rp {metrics['current_price']:,.0f}",
                    f"{metrics['daily_change']:+,.0f} ({metrics['daily_change_pct']:+.2f}%)"
                )
            
            with col2:
                st.metric("52W High", f"Rp {metrics['high_52w']:,.0f}")
            
            with col3:
                st.metric("52W Low", f"Rp {metrics['low_52w']:,.0f}")
            
            with col4:
                st.metric("Volatility", f"{metrics['volatility']:.1f}%")
            
            # Price chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="Price"
            ))
            
            fig.update_layout(
                title=f"{selected_stock_name} - Price Chart",
                xaxis_rangeslider_visible=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Failed to fetch stock data")
    
    with tab2:
        st.subheader(f"🤖 Price Prediction for {selected_stock_name}")
        
        if st.button("🚀 Generate Prediction", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    # Clean stock symbol for prediction
                    stock_symbol = selected_stock_code.replace('.JK', '').replace('^', '')
                    
                    # Generate predictions
                    predictions = predictor.predict_prices(stock_symbol, days_ahead=prediction_days)
                    
                    if predictions is not None:
                        st.success("✅ Prediction completed!")
                        
                        # Create prediction dataframe
                        future_dates = pd.date_range(
                            start=stock_data.index[-1] + pd.Timedelta(days=1),
                            periods=prediction_days,
                            freq='D'
                        )
                        
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Price': predictions
                        })
                        
                        # Display predictions table
                        st.dataframe(
                            pred_df.style.format({'Predicted_Price': 'Rp {:,.0f}'}),
                            use_container_width=True
                        )
                        
                        # Prediction chart
                        fig_pred = go.Figure()
                        
                        # Historical prices (last 30 days)
                        fig_pred.add_trace(go.Scatter(
                            x=stock_data.index[-30:],
                            y=stock_data['Close'].iloc[-30:],
                            mode='lines',
                            name='Historical Price',
                            line=dict(color='blue')
                        ))
                        
                        # Predicted prices
                        fig_pred.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_pred.update_layout(
                            title=f"Price Prediction - {prediction_days} Days Ahead",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                    else:
                        st.error("❌ Prediction failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    with tab3:
        st.subheader(f"💰 Valuation Analysis for {selected_stock_name}")
        
        if st.button("📊 Generate Valuation Report", type="primary"):
            with st.spinner("Analyzing with AI..."):
                try:
                    current_price = stock_data['Close'].iloc[-1]
                    
                    # Get predictions first
                    stock_symbol = selected_stock_code.replace('.JK', '').replace('^', '')
                    predictions = predictor.predict_prices(stock_symbol, days_ahead=7)
                    
                    if predictions:
                        # Generate valuation analysis
                        analysis = valuation_analyzer.analyze_stock_valuation(
                            selected_stock_name,
                            current_price,
                            predictions,
                            stock_data
                        )
                        
                        st.success("✅ Valuation analysis completed!")
                        st.markdown(analysis)
                    else:
                        st.error("❌ Unable to generate predictions for valuation")
                        
                except Exception as e:
                    st.error(f"Error during valuation analysis: {str(e)}")

else:
    st.error("❌ Failed to initialize application components. Please check your model files and configuration.")

# Footer
st.markdown("---")
st.markdown("*Powered by Deep Learning & Azure OpenAI | Data from Yahoo Finance*")
