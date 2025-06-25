import os

# Indonesian Stocks Configuration
INDONESIAN_STOCKS = {
    "Bank BCA (BBCA)": "BBCA.JK",
    "Astra International (ASII)": "ASII.JK", 
    "PT Indofood Sukses Makmur Tbk (INDF)": "INDF.JK",
    "Telkom Indonesia (TLKM)": "TLKM.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
    "Bank Negara Indonesia (BBNI)": "BBNI.JK",
    "Index Saham Indonesia (IHSG)": "^JKSE"
}

# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
    'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
    'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4'),
    'api_version': '2024-02-01'
}

# Model Paths
MODEL_PATHS = {
    'model': 'models/indonesian_stock_prediction_model.h5',
    'scalers': 'models/stock_scalers.pkl',
    'config': 'models/model_config.pkl'
}

# App Settings
APP_SETTINGS = {
    'max_prediction_days': 30,
    'default_prediction_days': 7,
    'data_period': '2y',
    'sequence_length': 60
}
