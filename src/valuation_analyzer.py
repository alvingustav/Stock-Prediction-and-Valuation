import openai
from azure.identity import DefaultAzureCredential
import json
from config import AZURE_OPENAI_CONFIG

class ValuationAnalyzer:
    def __init__(self):
        self.client = None
        self._setup_openai_client()
    
    def _setup_openai_client(self):
        """Setup Azure OpenAI client"""
        try:
            openai.api_type = "azure"
            openai.api_base = AZURE_OPENAI_CONFIG['endpoint']
            openai.api_key = AZURE_OPENAI_CONFIG['api_key']
            openai.api_version = AZURE_OPENAI_CONFIG['api_version']
            self.client = openai
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
    
    def analyze_stock_valuation(self, stock_name, current_price, predicted_prices, historical_data=None):
        """Analyze stock valuation using GPT-4"""
        if not self.client:
            return "Azure OpenAI not configured properly"
        
        try:
            # Prepare analysis prompt
            prompt = self._create_valuation_prompt(
                stock_name, current_price, predicted_prices, historical_data
            )
            
            # Call Azure OpenAI
            response = self.client.ChatCompletion.create(
                engine=AZURE_OPENAI_CONFIG['deployment_name'],
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in Indonesian stock market."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error in valuation analysis: {str(e)}"
    
    def _create_valuation_prompt(self, stock_name, current_price, predicted_prices, historical_data):
        """Create prompt for valuation analysis"""
        avg_predicted = sum(predicted_prices) / len(predicted_prices)
        price_trend = "increasing" if avg_predicted > current_price else "decreasing"
        
        prompt = f"""
        Analyze the valuation and investment outlook for {stock_name}:
        
        Current Price: Rp {current_price:,.0f}
        7-day Average Predicted Price: Rp {avg_predicted:,.0f}
        Price Trend: {price_trend}
        
        Please provide:
        1. Valuation assessment (Undervalued/Fair/Overvalued)
        2. Investment recommendation (Buy/Hold/Sell)
        3. Key risks and opportunities
        4. Target price range
        5. Time horizon recommendation
        
        Focus on Indonesian market context and provide concise, actionable insights.
        Use recent data.
        """
        
        return prompt
