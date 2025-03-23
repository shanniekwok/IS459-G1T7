from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, lit
import numpy as np
import joblib
import findspark
import os
import pandas as pd
import math
import boto3

findspark.init()

# Create Spark session
spark = SparkSession.builder \
    .appName("EnergyPrediction") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

# Load Parquet file
s3_path = "s3a://is459-g1t7-smart-meters-in-london/processed-data/merged_df1_df3_df7_df8"
df_spark = spark.read.parquet(s3_path)

# Define features
features = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "pressure", "humidity", "cloudCover", "windSpeed", "windBearing", 
    "precipType"
]

target = "energy_mean"

# Handle missing values in Spark DataFrame
df_spark = df_spark.dropna(subset=["energy_sum"])

# Handle precipType in Spark
if "precipType" in df_spark.columns:
    df_spark = df_spark.fillna({"precipType": "none"})
    
    # Get unique precipType values to encode
    precip_types = [row['precipType'] for row in df_spark.select('precipType').distinct().collect()]
    
    # Create a mapping dictionary for precipitation types
    precip_map = {val: idx for idx, val in enumerate(precip_types)}
    
    # Apply the mapping using Spark functions
    mapping_expr = None
    for val, idx in precip_map.items():
        if mapping_expr is None:
            mapping_expr = when(col("precipType") == val, lit(idx))
        else:
            mapping_expr = mapping_expr.when(col("precipType") == val, lit(idx))
    
    df_spark = df_spark.withColumn("precipType", mapping_expr)

# Calculate the total number of rows
total_rows = df_spark.count()
print(f"Total rows: {total_rows}")

# Split into train and test in Spark
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)

# Save test data count for later evaluation
test_count = test_df.count()
print(f"Test set size: {test_count}")

# Process training data in batches
batch_size = 100000  # Adjust based on your memory constraints
num_batches = math.ceil(train_df.count() / batch_size)
print(f"Processing training data in {num_batches} batches")

# Initialize an empty model or train on the first batch
rf_model = None
first_batch = True

for i in range(num_batches):
    print(f"Processing batch {i+1}/{num_batches}")
    
    # Get the current batch
    current_batch = train_df.limit(batch_size) if i == 0 else train_df.subtract(train_df.limit(i * batch_size)).limit(batch_size)
    
    # Convert current batch to pandas
    batch_pandas = current_batch.select(features + [target]).toPandas()
    
    if not batch_pandas.empty:
        X_batch = batch_pandas[features]
        y_batch = batch_pandas[target]
        
        if first_batch:
            # Initialize and train the model on first batch
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X_batch, y_batch)
            first_batch = False
        else:
            # Warm start for subsequent batches
            rf_model_batch = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1, 
                warm_start=True
            )
            # Initialize with the previous model's estimators
            rf_model_batch.estimators_ = rf_model.estimators_.copy()
            rf_model_batch.fit(X_batch, y_batch)
            
            # Update the model
            rf_model = rf_model_batch

# Evaluate on test data using batching
print("Evaluating on test data...")
batch_size_test = 50000  # Smaller batch for testing
num_test_batches = math.ceil(test_count / batch_size_test)

all_predictions = []
all_actuals = []

for i in range(num_test_batches):
    print(f"Processing test batch {i+1}/{num_test_batches}")
    
    # Get the current test batch
    current_test_batch = test_df.limit(batch_size_test) if i == 0 else test_df.subtract(test_df.limit(i * batch_size_test)).limit(batch_size_test)
    
    # Convert current batch to pandas
    test_batch_pandas = current_test_batch.select(features + [target]).toPandas()
    
    if not test_batch_pandas.empty:
        X_test_batch = test_batch_pandas[features]
        y_test_batch = test_batch_pandas[target]
        
        # Make predictions
        batch_preds = rf_model.predict(X_test_batch)
        
        # Store predictions and actuals
        all_predictions.extend(batch_preds.tolist())
        all_actuals.extend(y_test_batch.tolist())

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(all_actuals, all_predictions))
print(f"RF RMSE: {rmse}")

# Display a sample of predictions
sample_size = min(20, len(all_predictions))
predictions_df = pd.DataFrame({"Actual": all_actuals[:sample_size], "Predicted": all_predictions[:sample_size]})
print(predictions_df)

# Save the trained Random Forest model
repo_root = os.getcwd()  
model_dir = os.path.join(repo_root, "random_forest_model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "randomforest.pkl")
joblib.dump(rf_model, model_path)

print(f"Random Forest model saved successfully at: {model_path}")

# Upload the model to S3
s3 = boto3.resource('s3')
bucket_name = 'is459-g1t7-smart-meters-in-london'
key = "ml-models/randomforest.pkl"

# Upload the file
s3.Bucket(bucket_name).upload_file(model_path, key)

print(f"Random Forest model saved successfully at: s3://{bucket_name}/{key}")

# Stop Spark Session
spark.stop()