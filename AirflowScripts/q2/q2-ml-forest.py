from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql import SparkSession
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
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

# Add this right after creating your Spark session
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

# Load Parquet file
s3_path = "s3a://is459-g1t7-smart-meters-in-london/processed-data/merged_daily_weather_data"
df_spark = spark.read.parquet(s3_path)

# Define features
features = [
    "is_weekend", "temp_daylight_interaction", "is_holiday",
    "temperaturemax", "apparenttemperaturemax", "pressure",
    "apparenttemperaturemin", "apparenttemperaturehigh",
    "temperaturehigh", "daylight_duration"
]

target = "energy_mean"

# Handle missing values in Spark DataFrame
df_spark = df_spark.dropna(subset=[target])

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
s3.Bucket(bucket_name).upload_file(model_path, key)
print(f"Random Forest model saved successfully at: s3://{bucket_name}/{key}")

print("\n📦 Generating Actual vs Predicted CSVs for selected LCLids...")

# Re-evaluate test set with LCLid
test_pd_with_lclid = test_df.select(["lclid", "date"] + features + [target]).dropna().toPandas()
X_full = test_pd_with_lclid[features]
y_full = test_pd_with_lclid[target]
lclids = test_pd_with_lclid["lclid"]

y_full_preds = rf_model.predict(X_full)

full_predictions_df = pd.DataFrame({
    "LCLid": lclids,
    "Date": test_pd_with_lclid["date"],
    "Actual": y_full.values,
    "Predicted": y_full_preds
})

# LCLids for visualisation

selected_lclids = {"MAC000002", "MAC000030", "MAC000040", "MAC000103", "MAC000110",
                    "MAC000112", "MAC000246", "MAC000323", "MAC000339", "MAC000379",
                    "MAC000386", "MAC000450", "MAC000464", "MAC000487", "MAC000535",
                    "MAC000557", "MAC000569", "MAC000593", "MAC000609", "MAC000713",
                    "MAC000768", "MAC000778", "MAC000797", "MAC000816", "MAC000850",
                    "MAC000886", "MAC000902", "MAC000948", "MAC000974", "MAC001145",
                    "MAC001239", "MAC001251", "MAC001271", "MAC001510", "MAC001528",
                    "MAC001533", "MAC001628", "MAC001653", "MAC001689", "MAC001710",
                    "MAC001734", "MAC001736", "MAC001776", "MAC001819", "MAC001836",
                    "MAC001843", "MAC001893", "MAC001989", "MAC002025", "MAC002068",
                    "MAC002113", "MAC002134", "MAC002150", "MAC002199", "MAC002249",
                    "MAC002260", "MAC002314", "MAC002552", "MAC002562", "MAC002563",
                    "MAC002601", "MAC002613", "MAC002628", "MAC002813", "MAC002924",
                    "MAC002937", "MAC002959", "MAC003072", "MAC003110", "MAC003166",
                    "MAC003182", "MAC003196", "MAC003211", "MAC003212", "MAC003221",
                    "MAC003223", "MAC003239", "MAC003252", "MAC003257", "MAC003281",
                    "MAC003286", "MAC003305", "MAC003348", "MAC003388", "MAC003394",
                    "MAC003400", "MAC003422", "MAC003423", "MAC003428", "MAC003449",
                    "MAC003463", "MAC003474", "MAC003482", "MAC003536", "MAC003553",
                    "MAC003557", "MAC003566", "MAC003579", "MAC003597", "MAC003606",
                    "MAC003613", "MAC003634", "MAC003646", "MAC003656", "MAC003668",
                    "MAC003680", "MAC003683", "MAC003686", "MAC003718", "MAC003719",
                    "MAC003737", "MAC003740", "MAC003775", "MAC003805", "MAC003807",
                    "MAC003817", "MAC003826", "MAC003840", "MAC003844", "MAC003851",
                    "MAC003856", "MAC003863", "MAC003874", "MAC003895", "MAC004034",
                    "MAC004054", "MAC004179", "MAC004247", "MAC004258", "MAC004319",
                    "MAC004374", "MAC004387", "MAC004431", "MAC004436", "MAC004529",
                    "MAC004543", "MAC004554", "MAC004593", "MAC004633", "MAC004713",
                    "MAC004789", "MAC004832", "MAC004872", "MAC004900", "MAC004955",
                    "MAC004988", "MAC004997", "MAC005042", "MAC005062", "MAC005159",
                    "MAC005160", "MAC005172", "MAC005283", "MAC005406", "MAC005421",
                    "MAC005468", "MAC005523"}
s3_prefix = "quicksight-folder/q2-query"

for lclid in selected_lclids:
    filtered = full_predictions_df[full_predictions_df["LCLid"] == lclid]
    if filtered.empty:
        print(f"No data for {lclid}")
        continue

    local_path = os.path.join(repo_root, f"{lclid}-Actual-vs-Predicted.csv")
    filtered.to_csv(local_path, index=False)

    s3_key = f"{s3_prefix}/{lclid}-Actual-vs-Predicted.csv"
    try:
        s3.Bucket(bucket_name).upload_file(local_path, s3_key)
        print(f"Uploaded: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload {lclid}: {e}")
    finally:
        os.remove(local_path)

# Stop Spark Session
spark.stop()