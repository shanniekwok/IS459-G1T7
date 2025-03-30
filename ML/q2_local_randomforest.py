from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import joblib
import findspark
import os
import math

# Initialize Spark
findspark.init()
spark = SparkSession.builder \
    .appName("EnergyPrediction") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

# Load local CSV via Spark
repo_root = os.getcwd()
csv_path = os.path.join(repo_root, "q2_cleaned_data.csv")
df_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path).limit(200_000)

# Define feature columns
features = [
    "temp_daylight_interaction", "is_weekend", "pressure",
    "temperaturemax", "temperaturemin", "windbearing", "windspeed", "humidity", "cloudcover"
]

target = "energy_mean"

# Drop rows with missing target
df_spark = df_spark.dropna(subset=[target])

# Count rows
total_rows = df_spark.count()
print(f"Total rows: {total_rows}")

# Train-test split
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)
test_count = test_df.count()
print(f"Test set size: {test_count}")

# Process training data in batches
batch_size = 50000
num_batches = math.ceil(train_df.count() / batch_size)
print(f"Processing training data in {num_batches} batches")

rf_model = None
first_batch = True

for i in range(num_batches):
    print(f"Processing batch {i + 1}/{num_batches}")
    
    # Use offset-based limit for batching
    current_batch = train_df.limit(batch_size) if i == 0 else train_df.subtract(train_df.limit(i * batch_size)).limit(batch_size)
    batch_pandas = current_batch.select(features + [target]).dropna().toPandas()

    if not batch_pandas.empty:
        X_batch = batch_pandas[features]
        y_batch = batch_pandas[target]
        
        if first_batch:
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X_batch, y_batch)
            first_batch = False
        else:
            rf_model_batch = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1,
                warm_start=True
            )
            rf_model_batch.estimators_ = rf_model.estimators_.copy()
            rf_model_batch.fit(X_batch, y_batch)
            rf_model = rf_model_batch

# Evaluate on test data in batches
print("Evaluating on test data...")
batch_size_test = 50_000
num_test_batches = math.ceil(test_count / batch_size_test)

all_predictions = []
all_actuals = []

for i in range(num_test_batches):
    print(f"Processing test batch {i + 1}/{num_test_batches}")
    
    current_test_batch = test_df.limit(batch_size_test) if i == 0 else test_df.subtract(test_df.limit(i * batch_size_test)).limit(batch_size_test)
    test_batch_pandas = current_test_batch.select(features + [target]).dropna().toPandas()
    
    if not test_batch_pandas.empty:
        X_test_batch = test_batch_pandas[features]
        y_test_batch = test_batch_pandas[target]
        batch_preds = rf_model.predict(X_test_batch)
        
        all_predictions.extend(batch_preds.tolist())
        all_actuals.extend(y_test_batch.tolist())

# RMSE Evaluation
rmse = math.sqrt(mean_squared_error(all_actuals, all_predictions))
print(f"Random Forest RMSE: {rmse:.4f}")

# Show top 20 predictions
sample_size = min(20, len(all_predictions))
predictions_df = pd.DataFrame({
    "Actual": all_actuals[:sample_size],
    "Predicted": all_predictions[:sample_size]
})
print(predictions_df)

repo_root = os.getcwd()  
model_dir = os.path.join(repo_root, "random_forest_model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "randomforest.pkl")
joblib.dump(rf_model, model_path)
print(f"Model saved at: {model_path}")

# import boto3
# s3 = boto3.resource('s3')
# bucket_name = 'is459-g1t7-smart-meters-in-london'
# key = "ml-models/randomforest.pkl"
# s3.Bucket(bucket_name).upload_file(model_path, key)
# print(f"Model uploaded to s3://{bucket_name}/{key}")

# ========================================================

# Stop Spark
spark.stop()