import random

import joblib
import pandas as pd

def prepareModel(modelFile):
    return joblib.load(modelFile)

def prepareData():
    df = pd.read_csv("yellow_tripdata_2016-01.csv")
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    #df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 50)]  # Trips under 50 miles
    #df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 120)]
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
    return df

def getTime(model, dropoff_latitude, dropoff_longitude):

    features = [
        #'trip_distance',
        'passenger_count',
        'pickup_hour',
        'pickup_weekday',
        'is_weekend',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ]

    df = prepareData()
    input = df[features].head(1)
    input['dropoff_latitude'] = dropoff_latitude
    input['dropoff_longitude'] = dropoff_longitude

    print(input['pickup_latitude'])
    print(input['pickup_longitude'])

    return model.predict(input)

model = prepareModel('taxi_duration_model_noDistance.pkl')
print(getTime(model, 40.5, -74.0))
print(getTime(model, 40.6, -74.0))
print(getTime(model, 40.7, -74.0))
print(getTime(model, 40.8, -74.0))
print(getTime(model, 40.9, -74.0))

