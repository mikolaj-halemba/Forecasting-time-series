import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.forecast import Forecast

best_param = {
    'order': (1, 1, 1),
    'seasonal_order': (0, 0, 0, 0),
    'trend': 'c'
}

data = {
    'year': [2011, 2011, 2011, 2011],
    'quarter': [1, 2, 3, 4],
    'volume_sales': [100, 200, 300, 400]
}

df = pd.DataFrame(data)
df.index = pd.PeriodIndex(year=df['year'], quarter=df['quarter'], freq='Q')

@pytest.fixture
def forecast_instance():
    return Forecast(best_param)

@patch('src.forecast.ARIMA')
def test_transform(mock_arima, forecast_instance):
    mock_model_instance = MagicMock()
    mock_arima.return_value = mock_model_instance
    mock_fit_instance = MagicMock()
    mock_model_instance.fit.return_value = mock_fit_instance
    mock_fit_instance.forecast.return_value = pd.Series([500, 600, 700, 800], index=pd.period_range(start=df.index[-1] + 1, periods=4, freq='Q'))

    forecast = forecast_instance.transform(df)

    assert not forecast.empty
    assert list(forecast) == [500, 600, 700, 800]
    mock_arima.assert_called_once_with(df['volume_sales'], **best_param)
    mock_model_instance.fit.assert_called_once()
    mock_fit_instance.forecast.assert_called_once_with(steps=12)

@patch('src.forecast.ARIMA')
def test_forecast(mock_arima, forecast_instance):
    mock_model_instance = MagicMock()
    mock_arima.return_value = mock_model_instance
    mock_fit_instance = MagicMock()
    mock_model_instance.fit.return_value = mock_fit_instance
    mock_fit_instance.forecast.return_value = pd.Series([500, 600, 700, 800], index=pd.period_range(start=df.index[-1] + 1, periods=4, freq='Q'))

    forecast = forecast_instance._forecast(df)

    assert not forecast.empty
    assert list(forecast) == [500, 600, 700, 800]
    mock_arima.assert_called_once_with(df['volume_sales'], **best_param)
    mock_model_instance.fit.assert_called_once()
    mock_fit_instance.forecast.assert_called_once_with(steps=12)

