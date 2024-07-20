from src.base_operation import BaseOperation
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from utils.utils import log_execution_time


class Forecast(BaseOperation):
    def __init__(self, best_param: Dict[str, Any]) -> None:
        """
        Initializes the Forecast class with the given ARIMA model parameters.

        :param best_param: Dictionary containing the best parameters for the ARIMA model.
        """
        self.best_param: Dict[str, Any] = best_param

    @log_execution_time
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by forecasting future sales.

        :param df: Input DataFrame containing sales data.
        :return: DataFrame with the forecasted sales data.
        """
        df = self._forecast(df)
        return df

    def _forecast(self, df: pd.DataFrame):
        """
        Performs the forecasting using the ARIMA model.

        :param df: Input DataFrame containing sales data.
        :return: Series with the forecasted sales data.
        """
        sales_series = df["volume_sales"]
        sales_series.index = pd.PeriodIndex(
            year=df["year"], quarter=df["quarter"], freq="Q"
        )
        best_model = ARIMA(sales_series, **self.best_param)
        best_model_fit = best_model.fit()
        forecast = best_model_fit.forecast(steps=12)
        # self._plot_forecast(sales_series, forecast)

        return forecast

    def _plot_forecast(self, sales_series: pd.Series, forecast: pd.Series):
        """
        Plots the historical sales data along with the forecasted sales data.

        :param sales_series: Series containing the historical sales data.
        :param forecast: Series containing the forecasted sales data.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            sales_series.index.to_timestamp(), sales_series, label="Historical Data"
        )
        plt.plot(
            forecast.index.to_timestamp(), forecast, label="Forecast", color="orange"
        )
        plt.title("Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales Volume")
        plt.legend()
        plt.grid(True)
        plt.show()
