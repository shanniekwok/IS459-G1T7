from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession

import joblib
import findspark
import os
import pandas as pd

findspark.init()

# Create Spark session
spark = SparkSession.builder.appName("EnergyPrediction").config("spark.driver.memory", "12g").config("spark.executor.memory", "6g").getOrCreate()

# Load Parquet file
repo_root = os.getcwd()  
parquet_path = os.path.join(repo_root, "merged_df1_df3_df7_df8")
df_spark = spark.read.parquet(parquet_path).limit(300_000)  # Load only 500k rows

# **Convert to Pandas (Much Faster Now)**
df = df_spark.toPandas()

#encoding precipitation type
if "precipType" in df.columns:
    df = df.fillna({"precipType": "none"})
le = LabelEncoder()
df["precipType"] = le.fit_transform(df["precipType"])

features = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "pressure", "humidity", "cloudCover", "windSpeed", "windBearing", 
    "precipType"
]

target = "energy_mean"

# **Drop rows with missing values**
df = df.dropna(subset=["energy_sum"])  

# **Splitting dataset into training & testing sets**
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Train Random Forest Regressor**
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# **Evaluate**
y_pred = rf_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RF RMSE: {rmse}")

# **Convert predictions to DataFrame for easy display**
predictions_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(predictions_df.head(20))

# Save the trained Random Forest model
repo_root = os.getcwd()  
model_dir = os.path.join(repo_root, "random_forest_model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "randomforest.pkl")
joblib.dump(rf_model, model_path)

print(f"Random Forest model saved successfully at: {model_path}")

# Stop Spark Session
spark.stop()