import pandas as pd
import numpy as np
from ols import OLS

REFERENCE_YEAR = 2025
# I wanted to predict my car's price for this example
PREDICT_CAR = pd.DataFrame({'Make': ['Toyota'], 'Age': [15], 'Kilometer': [150000], 'Color': ['Silver']})

# Import data
cars_df = pd.read_csv('cars.csv')

###################
## Data cleaning ##
###################

cars_df = cars_df.dropna()
cars_df['Age'] = REFERENCE_YEAR - cars_df['Year']

# Separate into prediction and response variables
X = cars_df[['Make', 'Age', 'Kilometer', 'Color']]
y = cars_df['Price']

# Add PREDICT_CAR to first row, then encode X into numerical matrix
X = pd.concat([PREDICT_CAR, X], ignore_index=True)
factors = ['Make', 'Color']
X = pd.get_dummies(X, columns=factors, dtype=int)

# Extract encoded PREDICT_CAR
predict_car_encoded = X.iloc[0:1].copy()
X = X.iloc[1:]


print(X.dtypes)

# Convert to numpy array
X_matrix = X.to_numpy()
y_array = y.to_numpy()

ols = OLS()
ols.fit(X_matrix, y_array)
print(ols.get_weights())

# Predict the price of a new car
print(ols.predict(predict_car_encoded.to_numpy()[0]))

