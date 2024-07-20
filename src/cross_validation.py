from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from src.base_operation import BaseOperation
from src.load_model import ModelLoader
from utils.utils import log_results
from utils.utils import log_execution_time


class TimeSeriesCrossValidator(BaseOperation):
    def __init__(self, param_grid: Dict[str, Any], n_splits: int = 5) -> None:
        """
        Initializes TimeSeriesCrossValidator with the given parameter grid and number of splits.

        :param param_grid: Dictionary with parameters for GridSearchCV.
        :param n_splits: Number of splits for TimeSeriesSplit.
        """
        self.logs_path = param_grid["logs"]["path"]
        self.model_path = param_grid["model"]["path"]
        self.model_loader = ModelLoader(self.model_path)
        self.param_grid = param_grid["param_grid"]
        self.metrics = param_grid["evaluation"]["metrics"]
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = 0.0

    @log_execution_time
    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transforms data and tunes hyperparameters.

        :param df: DataFrame with input data.
        :return: Best parameters and score.
        """
        X = df.index.values.reshape(-1, 1)
        y = df["volume_sales"].values
        best_params, _ = self._tune_parameters(X, y)
        return best_params

    @log_results()
    def _tune_parameters(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tunes parameters using GridSearchCV.

        :param X: Input data.
        :param y: Output data.
        :return: Best parameters and score.
        """
        grid_search = GridSearchCV(
            estimator=self.model_loader,
            param_grid=self.param_grid,
            cv=self.tscv,
            scoring=self.metrics,
        )
        grid_search.fit(X, y)
        self.best_params, _ = grid_search.best_params_, grid_search.best_score_

        return self.best_params, _
