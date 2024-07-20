import pandas as pd
from typing import Dict, Any,Tuple
from statsmodels.tsa.stattools import adfuller
from src.base_operation import BaseOperation
from utils.utils import log_execution_time
from utils.utils import logger


class DataPreprocessor(BaseOperation):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes DataPreprocessor with the given configuration.

        :param config: Dictionary containing configuration parameters.
        """
        self.raw_data_path: str = config['data']['raw_path']

    @log_execution_time
    def transform(self) -> pd.DataFrame:
        """
        Transforms the raw data by reading, preprocessing, aggregating, and making it stationary.

        :return: Transformed and stationary DataFrame.
        """
        df_sales = self._read_data()
        df_sales_preprocessed = self._preprocess_data(df_sales)
        df_sales_prepared = self._aggregate_data(df_sales_preprocessed)
        df_sales_prepared, num_diffs = self._make_stationary(df_sales_prepared.copy(), 'volume_sales')
        return df_sales_prepared

    def _read_data(self) -> pd.DataFrame:
        """
        Reads raw sales data from the specified file path.

        :return: DataFrame containing raw sales data.
        """
        df_sales: pd.DataFrame = pd.read_csv(self.raw_data_path)
        return df_sales

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the raw data by renaming columns and filtering specific products.

        :param df: DataFrame containing raw sales data.
        :return: Preprocessed DataFrame with selected products.
        """
        df_sales_renamed = df.rename(columns={"volumeSales": "volume_sales"})
        df_filtered = df_sales_renamed[df_sales_renamed['product'].isin(['P1', 'P2', 'P3'])]
        return df_filtered

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the sales data by year and quarter.

        :param df: DataFrame containing preprocessed sales data.
        :return: Aggregated DataFrame with summed sales volumes.
        """
        df_agg_c1 = df.groupby(['year', 'quarter']).agg({'volume_sales': 'sum'}).reset_index()
        df_agg_c1.set_index(pd.PeriodIndex(year=df_agg_c1.year, quarter=df_agg_c1.quarter, freq='Q'), inplace=True)
        df_agg_c1['volume_sales'] = df_agg_c1['volume_sales']
        return df_agg_c1

    def _check_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Augmented Dickey-Fuller test to check the stationarity of the data.

        Parameters:
        timeseries (pd.Series): Time series data to be tested.

        Returns:
        bool: True if the data is stationary, False otherwise.
        """
        return adfuller(df)[1] <= 0.05

    def _make_stationary(self, df, column='volume_sales', max_diff=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Makes the data stationary by differencing.

        :param df: DataFrame containing the time series data.
        :param column: Column name of the time series data.
        :param max_diff: Maximum number of differences to apply.
        :return: Tuple containing the DataFrame with stationary data and the number of differences applied.
        """
        diff_count = 0
        while diff_count < max_diff:
            if self._check_stationarity(df[column]):
                logger.info(f"Data is stationary after {diff_count} differencing.")
                return df, diff_count
            df[column] = df[column].diff().dropna()
            diff_count += 1
        if not self._check_stationarity(df[column]):
            raise ValueError("Data is not stationary after maximum allowed differencing.")
        logger.info(f"Data is stationary after {diff_count} differencing.")
        return df, diff_count

