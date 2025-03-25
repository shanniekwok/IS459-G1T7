from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, lit
import numpy as np
import joblib
import os
import pandas as pd
import math
import boto3
import traceback
import sys

# Create Spark session - removed findspark dependency
print("Creating Spark session...")
spark = SparkSession.builder \
    .appName("EnergyPrediction") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.hadoop.hive.metastore.client.factory.class", "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory") \
    .config("spark.sql.catalogImplementation", "hive") \
    .enableHiveSupport() \
    .getOrCreate()

print("Spark session created successfully!")

# Define Glue database and table names
glue_database = "q2-processed-data-output"
glue_table = "merged_df1_df3_df7_df8"
s3_fallback_path = "s3a://is459-g1t7-smart-meters-in-london/processed-data/merged_df1_df3_df7_df8"

# Try multiple methods to load the data
print(f"Attempting to load data from Glue catalog: {glue_database}.{glue_table}")
df_spark = None

try:
    # Method 1: Using Glue catalog directly
    df_spark = spark.read.table(f"{glue_database}.{glue_table}")
    print("Successfully loaded data from Glue catalog using table()!")
except Exception as e:
    print(f"Error loading using table(): {str(e)}")
    print("Trying alternative method...")
    
    try:
        # Method 2: Using format parquet with catalog options
        df_spark = spark.read.format("parquet") \
            .option("catalogProvider", "glue") \
            .option("database", glue_database) \
            .option("table", glue_table) \
            .load()
        print("Successfully loaded data using format parquet with catalog options!")
    except Exception as e:
        print(f"Error loading with catalog options: {str(e)}")
        print("Trying direct S3 access...")
        
        try:
            # Method 3: Direct S3 path
            df_spark = spark.read.parquet(s3_fallback_path)
            print(f"Successfully loaded data directly from S3: {s3_fallback_path}")
        except Exception as e:
            print(f"Error loading directly from S3: {str(e)}")
            print("All loading methods failed. Exiting.")
            traceback.print_exc()
            sys.exit(1)

# Print schema and sample data for debugging
print("Data schema:")
df_spark.printSchema()
print("Sample data (5 rows):")
df_spark.show(5, truncate=False)

# Define features
features = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "pressure", "humidity", "cloudCover", "windSpeed", "windBearing", 
    "precipType"
]

target = "energy_mean"

# Verify columns exist in the dataframe
available_columns = set(df_spark.columns)
print(f"Available columns: {available_columns}")

missing_features = [f for f in features if f not in available_columns]
if missing_features:
    print(f"Warning: The following features are missing from the dataframe: {missing_features}")
    features = [f for f in features if f in available_columns]
    print(f"Proceeding with available features: {features}")

if target not in available_columns:
    print(f"Error: Target column '{target}' not found in dataframe. Available columns: {available_columns}")
    print("Checking if 'energy_sum' exists as an alternative...")
    if "energy_sum" in available_columns:
        print("Using 'energy_sum' instead of 'energy_mean'")
        target = "energy_sum"
    else:
        print("No suitable target column found. Exiting.")
        sys.exit(1)

# Handle missing values in Spark DataFrame
print("Handling missing values...")
df_spark = df_spark.dropna(subset=[target])
print(f"Rows after dropping NAs in target: {df_spark.count()}")

# Handle precipType in Spark
if "precipType" in df_spark.columns:
    print("Processing 'precipType' column...")
    df_spark = df_spark.fillna({"precipType": "none"})
    
    # Get unique precipType values to encode
    precip_types = [row['precipType'] for row in df_spark.select('precipType').distinct().collect()]
    print(f"Unique precipitation types: {precip_types}")
    
    # Create a mapping dictionary for precipitation types
    precip_map = {val: idx for idx, val in enumerate(precip_types)}
    
    # Apply the mapping using Spark functions
    mapping_expr = None
    for val, idx in precip_map.items():
        if mapping_expr is None:
            mapping_expr = when(col("precipType") == val, lit(idx))
        else:
            mapping_expr = mapping_expr.when(col("precipType") == val, lit(idx))
    
    # Add a default case for null values
    if mapping_expr is not None:
        mapping_expr = mapping_expr.otherwise(lit(0))
        df_spark = df_spark.withColumn("precipType", mapping_expr)

# Calculate the total number of rows
total_rows = df_spark.count()
print(f"Total rows: {total_rows}")

# Split into train and test in Spark
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)

# Save test data count for later evaluation
test_count = test_df.count()
print(f"Training set size: {train_df.count()}")
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
    batch_count = current_batch.count()
    print(f"Current batch size: {batch_count}")
    
    if batch_count == 0:
        print("Warning: Empty batch, skipping")
        continue
    
    # Convert current batch to pandas
    batch_pandas = current_batch.select(features + [target]).toPandas()
    
    if not batch_pandas.empty:
        X_batch = batch_pandas[features]
        y_batch = batch_pandas[target]
        
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        
        if first_batch:
            print("Training initial model...")
            # Initialize and train the model on first batch
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X_batch, y_batch)
            first_batch = False
            print("Initial model training completed")
        else:
            print("Updating model with new batch...")
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
            print("Model update completed")

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
    test_batch_count = current_test_batch.count()
    print(f"Current test batch size: {test_batch_count}")
    
    if test_batch_count == 0:
        print("Warning: Empty test batch, skipping")
        continue
    
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
if all_predictions and all_actuals:
    rmse = math.sqrt(mean_squared_error(all_actuals, all_predictions))
    print(f"RF RMSE: {rmse}")

    # Display a sample of predictions
    sample_size = min(20, len(all_predictions))
    predictions_df = pd.DataFrame({"Actual": all_actuals[:sample_size], "Predicted": all_predictions[:sample_size]})
    print(predictions_df)

    # Save the trained Random Forest model
    print("Saving model...")
    temp_dir = "/tmp/random_forest_model"
    os.makedirs(temp_dir, exist_ok=True)
    model_path = os.path.join(temp_dir, "randomforest.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Random Forest model saved locally at: {model_path}")

    # Upload the model to S3
    try:
        print("Uploading model to S3...")
        s3 = boto3.resource('s3')
        bucket_name = 'is459-g1t7-smart-meters-in-london'
        key = "ml-models/randomforest.pkl"

        # Upload the file
        s3.Bucket(bucket_name).upload_file(model_path, key)
        print(f"Random Forest model saved successfully at: s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Error uploading model to S3: {str(e)}")
        traceback.print_exc()
else:
    print("Error: No predictions were made. Check if test data was processed correctly.")

# Stop Spark Session
print("Stopping Spark session...")
spark.stop()
print("Job completed!")