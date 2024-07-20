from typing import Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

class ModelLoader(BaseEstimator, RegressorMixin):
    def __init__(self, file_path: str = None,
                 order: Tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: str = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True) -> None:
        """
        Initializes ModelLoader with the given file path and ARIMA order.

        :param file_path: Path to the model file.
        :param order: Order of the ARIMA model.
        :param seasonal_order: Seasonal order of the ARIMA model.
        :param trend: Trend parameter for the ARIMA model.
        :param enforce_stationarity: Whether to enforce stationarity in the model.
        :param enforce_invertibility: Whether to enforce invertibility in the model.
        """
        self.file_path = file_path
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Private method to load the model from the given path.

        Attempts to load a pre-trained model from the specified file path.
        Handles exceptions if the file is not found or cannot be unpickled.
        """
        try:
            with open(self.file_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
        except pickle.UnpicklingError:
            print(f"Error: The file at {self.file_path} could not be unpickled.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'ModelLoader':
        """
        Fits the ARIMA model to the given data.

        :param X: Input data features (not used in ARIMA but kept for compatibility with scikit-learn).
        :param y: Target variable, the time series to be modeled.
        :return: Self, with the fitted model.
        """
        self.model = ARIMA(y, order=self.order, trend=self.trend,
                           enforce_stationarity=self.enforce_stationarity,
                           enforce_invertibility=self.enforce_invertibility).fit()
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Makes predictions using the fitted ARIMA model.

        :param X: Input data features (used to determine the number of steps to forecast).
        :return: Predicted values for the given number of steps.
        """
        if self.model is None:
            raise ValueError("Model not fitted, cannot predict.")
        return self.model.forecast(steps=len(X))
