import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load the DataFrame from the .pkl file
df = pd.read_csv('troop_movements_1m.csv')


# load 'real' data and use the model to predict Empire or Resistance
# Read the raw data CSV file into a pandas DataFrame

# Filter out rows where 'unit type' is 'unknown'
# filtered_df = df[df['unit_type'] != 'invalid_unit']
# invalid_unit_df = df[df['unit_type'] == 'invalid_unit'] # Filter only invalid_units

df["unit_type"] = df["unit_type"].apply(lambda x: "unknown" if x == "invalid_unit" else x)

missing_location_x_df = df[df['location_x'].isna()]
df["location_x"] = missing_location_x_df["location_x"].apply(lambda x: 1 if x == "NaN" else x)


# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, np.nan, 4, np.nan, 6]})

# Use ffill to fill missing values
df_filled = df.ffill()




# Fill missing values in 'location_x' with the mode of the column
# location_x_mode = df['location_x'].mode()[0]
# df['location_x'].fillna(location_x_mode, inplace=True)

# Optionally, save the filtered DataFrame to a new CSV file
# filtered_df.to_csv('trained_model.csv', index=False)
# df10m = pd.read_csv('troop_movements_1m.csv')

# print(df10m.head())

print('before: ' + df.head(20))
print('------------------------------------------')
print('after: ' + df_filled.head(20))


# # Convert the DataFrame to an Arrow Table
# table = pa.Table.from_pandas(df)

# # Save the Arrow Table as a Parquet file
# pq.write_table(table, 'trained_data_parquet.parquet')


# Write the cleaned data to a Parquet file
df.to_parquet('cleaned_troop_movements1m.parquet', index=False)

print('Data cleaning complete.')

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load clean data from parquet file
df10m = pd.read_parquet('cleaned_troop_movements1m.parquet')
features = ['homeworld', 'unit_type']
new_data = df10m[features]  # Features

# Convert categorical features to numeric using one-hot encoding
new_data_encoded = pd.get_dummies(new_data)

# Assuming you have new data for prediction stored in a pandas DataFrame called 'new_data_encoded'
predictions = model.predict(new_data_encoded)

# You can use the predictions as desired
# print(predictions)

df['predictions'] = predictions
df.head()