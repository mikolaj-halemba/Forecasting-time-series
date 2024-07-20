import pandas as pd
import pickle

# Load sales data
sales_data = pd.read_csv(r'C:\Users\mikol\Repositories\Nestle\data\historical_sales_volume.csv')

# Load the serialized model
with open(r'C:\Users\mikol\Repositories\Nestle\models\arima_model.pkl', 'rb') as file:
    arima_model = pickle.load(file)

# Display the data and model to understand their structure
print(sales_data.head())
print(arima_model)
