import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import joblib

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv("yellow_tripdata_2016-01.csv")

print(df.head())

print(df.isnull().sum())

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

print(df["trip_distance"][10])

df["trip_distance"].hist(bins=50)
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Frequency")
plt.yscale('log')
plt.title("Distribution of Trip Distances")
plt.show()

df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 50)]  # Trips under 50 miles
df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 120)]

df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)

print("Data preparation completed. Ready for modeling.")

features = [
    'trip_distance',
    'passenger_count',
    'pickup_hour',
    'pickup_weekday',
    'is_weekend',
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude'
]

X = df[features].copy()
y = df['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=222)

print("Training Random Forest model")
rf_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=10,
    random_state=222,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Model Performance:")
print(f"Root Mean Squared Error: {rmse:.4f} minutes")
print(f"R-squared: {r2:.4f}")

# Basic feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model

joblib.dump(rf_model, 'taxi_duration_model.pkl')
print("Model saved as 'taxi_duration_model.pkl'")
