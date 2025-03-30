import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

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

print(df[df["trip_distance"] > 10000]["trip_distance"])