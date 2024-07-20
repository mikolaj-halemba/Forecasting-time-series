import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.cross_validation import TimeSeriesCrossValidator

param_grid = {
    "logs": {"path": "path/to/logs"},
    "model": {"path": "path/to/model.pkl"},
    "param_grid": {"order": [(1, 1, 1), (2, 1, 2)]},
    "evaluation": {"metrics": "neg_mean_squared_error"},
}

data = {
    "year": [2011, 2011, 2011, 2011],
    "quarter": [1, 2, 3, 4],
    "volume_sales": [100, 200, 300, 400],
}

df = pd.DataFrame(data)
df.index = pd.PeriodIndex(year=df["year"], quarter=df["quarter"], freq="Q")
df.drop(columns=["year", "quarter"], inplace=True)


@pytest.fixture
def cross_validator():
    return TimeSeriesCrossValidator(param_grid)


@patch("src.cross_validation.GridSearchCV")
@patch("src.cross_validation.ModelLoader")
def test_transform(mock_model_loader, mock_grid_search, cross_validator):
    mock_model_instance = MagicMock()
    mock_model_loader.return_value = mock_model_instance
    mock_grid_search_instance = MagicMock()
    mock_grid_search.return_value = mock_grid_search_instance
    mock_grid_search_instance.fit.return_value = None
    mock_grid_search_instance.best_params_ = {"order": (1, 1, 1)}
    mock_grid_search_instance.best_score_ = -100.0

    best_params = cross_validator.transform(df)

    assert best_params == {"order": (1, 1, 1)}
    mock_grid_search.assert_called_once()
    mock_grid_search_instance.fit.assert_called_once()


@patch("src.cross_validation.GridSearchCV")
@patch("src.cross_validation.ModelLoader")
def test_tune_parameters(mock_model_loader, mock_grid_search, cross_validator):
    mock_model_instance = MagicMock()
    mock_model_loader.return_value = mock_model_instance
    mock_grid_search_instance = MagicMock()
    mock_grid_search.return_value = mock_grid_search_instance
    mock_grid_search_instance.fit.return_value = None
    mock_grid_search_instance.best_params_ = {"order": (1, 1, 1)}
    mock_grid_search_instance.best_score_ = -100.0

    X = df.index.values.reshape(-1, 1)
    y = df["volume_sales"].values
    best_params, best_score = cross_validator._tune_parameters(X, y)

    assert best_params == {"order": (1, 1, 1)}
    assert best_score == -100.0
    mock_grid_search.assert_called_once()
    mock_grid_search_instance.fit.assert_called_once()
