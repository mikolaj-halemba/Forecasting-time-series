import os
from src.data_preprocessing import DataPreprocessor
from src.cross_validation import TimeSeriesCrossValidator
from src.forecast import Forecast
from utils.utils import load_config

if __name__ == "__main__":
    # Load configs
    config = load_config(os.path.join('config', 'config.yaml'))

    # Cross-validation & Forecasting
    processed_data = DataPreprocessor(config).transform()
    best_param = TimeSeriesCrossValidator(config).transform(processed_data)
    df_forecast = Forecast(best_param).transform(processed_data)

