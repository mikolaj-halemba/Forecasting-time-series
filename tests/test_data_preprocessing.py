import pytest
import pandas as pd
from unittest.mock import patch
from src.data_preprocessing import DataPreprocessor

config = {"data": {"raw_path": "path/to/raw_data.csv"}}

raw_data = pd.DataFrame(
    {
        "year": [2011, 2011, 2011, 2011],
        "quarter": [1, 2, 3, 4],
        "volumeSales": [100, 200, 300, 400],
        "product": ["P1", "P2", "P3", "P4"],
    }
)

processed_data = pd.DataFrame(
    {"year": [2011, 2011, 2011], "quarter": [1, 2, 3], "volume_sales": [100, 200, 300]}
)

aggregated_data = pd.DataFrame(
    {"year": [2011, 2011, 2011], "quarter": [1, 2, 3], "volume_sales": [100, 200, 300]}
)
aggregated_data.index = pd.PeriodIndex(
    year=aggregated_data.year, quarter=aggregated_data.quarter, freq="Q"
)

stationary_data = aggregated_data.copy()


@pytest.fixture
def data_preprocessor():
    return DataPreprocessor(config)


@patch("src.data_preprocessing.pd.read_csv")
def test_read_data(mock_read_csv, data_preprocessor):
    mock_read_csv.return_value = raw_data
    df = data_preprocessor._read_data()
    assert not df.empty
    assert list(df.columns) == ["year", "quarter", "volumeSales", "product"]


def test_preprocess_data(data_preprocessor):
    df = data_preprocessor._preprocess_data(raw_data)
    assert not df.empty
    assert "volume_sales" in df.columns
    assert all(df["product"].isin(["P1", "P2", "P3"]))


def test_aggregate_data(data_preprocessor):
    df = data_preprocessor._aggregate_data(processed_data)
    assert not df.empty
    assert "volume_sales" in df.columns
    assert isinstance(df.index, pd.PeriodIndex)


@patch("src.data_preprocessing.adfuller")
def test_check_stationarity(mock_adfuller, data_preprocessor):
    mock_adfuller.return_value = (None, 0.01)  # Mocking ADF test to be stationary
    assert data_preprocessor._check_stationarity(stationary_data["volume_sales"])
