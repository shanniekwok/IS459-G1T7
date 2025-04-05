from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, isnan, when, count
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import os
import boto3
import seaborn as sns
import findspark
import math
import joblib

findspark.init()

# Create Spark session with adequate memory
spark = SparkSession.builder \
    .appName("FeatureImportanceAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

# For parquet compatibility
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

# Load Parquet file
s3_path = "s3a://is459-g1t7-smart-meters-in-london/processed-data/merged_daily_weather_data"
df_spark = spark.read.parquet(s3_path)

# Define all features
weather_features = [
    'temperaturemax', 'temperaturemin', 'temperaturehigh', 'temperaturelow',
    'apparenttemperaturemax', 'apparenttemperaturemin', 'apparenttemperaturehigh', 'apparenttemperaturelow', 
    'windbearing', 'dewpoint', 'cloudcover', 'windspeed', 'pressure', 'preciptype', 'visibility', 'humidity', 'uvindex', 'moonphase'
]

derived_features = [
    'day_of_week', 'month', 'season', 'is_weekend', 'is_holiday',
    'temp_variation', 'temp_humidity_interaction', 'temp_cloudcover_interaction',
    'temp_uvindex_interaction', 'weekend_energy_interaction', 'holiday_energy_interaction',
    'daylight_duration', 'temp_daylight_interaction'
]

target = "energy_mean"

# All features combined
all_features = weather_features + derived_features

# Identify categorical features
categorical_features = ['preciptype', 'holiday', 'day_of_week', 'month', 'season']
numerical_features = [f for f in all_features if f not in categorical_features]

# Process categorical features
indexers = []
for categorical_feature in categorical_features:
    if categorical_feature in df_spark.columns:
        # Fill nulls with a placeholder value for categorical features
        df_spark = df_spark.fillna({categorical_feature: "unknown"})
        
        # Create StringIndexer for the categorical feature
        indexer = StringIndexer(
            inputCol=categorical_feature, 
            outputCol=f"{categorical_feature}_indexed",
            handleInvalid="keep"  # Handle any new categories in test data
        )
        indexers.append(indexer)

# Create a pipeline for preprocessing
preprocessing_stages = indexers.copy()

# Execute preprocessing pipeline
preprocessing_pipeline = Pipeline(stages=preprocessing_stages)
preprocessor = preprocessing_pipeline.fit(df_spark)
preprocessed_df = preprocessor.transform(df_spark)

# Determine the final feature columns for sklearn
final_feature_cols = numerical_features.copy()
for feature in categorical_features:
    if feature in df_spark.columns:
        final_feature_cols.append(f"{feature}_indexed")

# Split the data into training and testing sets
train_df, test_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining set size: {train_df.count()} rows")
print(f"Testing set size: {test_df.count()} rows")

# Batch processing setup
batch_size = 100000  # Adjust based on memory constraints
train_count = train_df.count()
num_batches = math.ceil(train_count / batch_size)
print(f"Processing training data in {num_batches} batches")

# Initialize model
rf_model = None
first_batch = True

# Process training data in batches
for i in range(num_batches):
    print(f"Processing batch {i+1}/{num_batches}")
    
    # Get the current batch
    current_batch = train_df.limit(batch_size) if i == 0 else train_df.subtract(train_df.limit(i * batch_size)).limit(batch_size)
    
    # Convert to pandas
    batch_pandas = current_batch.select(final_feature_cols + [target]).toPandas()
    
    if not batch_pandas.empty:
        X_batch = batch_pandas[final_feature_cols]
        y_batch = batch_pandas[target]
        
        if first_batch:
            # Initialize and train model on first batch
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
            # Initialize with previous model's estimators
            rf_model_batch.estimators_ = rf_model.estimators_.copy()
            rf_model_batch.fit(X_batch, y_batch)
            
            # Update model
            rf_model = rf_model_batch

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Create a mapping from features to their names
feature_names = []
for feature in final_feature_cols:
    if feature in numerical_features:
        feature_names.append(feature)
    else:
        # For categorical features, find the original feature name
        base_feature = feature.split('_indexed')[0]
        feature_names.append(base_feature)

# Create DataFrame with feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Group by feature name and sum importances (in case of duplicate features)
importance_df = importance_df.groupby('Feature').sum().reset_index()

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print top 20 features
print("\nTop 20 most important features:")
print(importance_df.head(20))

# Save feature importances to CSV
output_dir = os.path.join(os.getcwd(), "feature_importance_results")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "feature_importances.csv")
importance_df.to_csv(csv_path, index=False)
print(f"\nFeature importances saved to {csv_path}")

# Create visualization
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plot_path = os.path.join(output_dir, "feature_importances_plot.png")
plt.savefig(plot_path)
print(f"Feature importance plot saved to {plot_path}")

# Evaluate on test data using batching
print("Evaluating on test data...")
test_count = test_df.count()
batch_size_test = 50000  # Smaller batch for testing
num_test_batches = math.ceil(test_count / batch_size_test)

all_predictions = []
all_actuals = []

for i in range(num_test_batches):
    print(f"Processing test batch {i+1}/{num_test_batches}")
    
    # Get current test batch
    current_test_batch = test_df.limit(batch_size_test) if i == 0 else test_df.subtract(test_df.limit(i * batch_size_test)).limit(batch_size_test)
    
    # Convert to pandas
    test_batch_pandas = current_test_batch.select(final_feature_cols + [target]).toPandas()
    
    if not test_batch_pandas.empty:
        X_test_batch = test_batch_pandas[final_feature_cols]
        y_test_batch = test_batch_pandas[target]
        
        # Make predictions
        batch_preds = rf_model.predict(X_test_batch)
        
        # Store predictions and actuals
        all_predictions.extend(batch_preds.tolist())
        all_actuals.extend(y_test_batch.tolist())

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(all_actuals, all_predictions))
print(f"RMSE on test data: {rmse}")

# Display sample predictions
sample_size = min(20, len(all_predictions))
predictions_df = pd.DataFrame({"Actual": all_actuals[:sample_size], "Predicted": all_predictions[:sample_size]})
print(predictions_df)

# Save the model
model_dir = os.path.join(output_dir, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rf_feature_importance.pkl")
joblib.dump(rf_model, model_path)
print(f"\nModel saved to {model_path}")

# Select top features for future use
top_n = 15  # Adjust as needed
top_features = importance_df['Feature'].head(top_n).tolist()

print(f"\nTop {top_n} features for future use:")
print(top_features)

# Save top features to text file
top_features_path = os.path.join(output_dir, "top_features.txt")
with open(top_features_path, 'w') as f:
    for feature in top_features:
        f.write(f"{feature}\n")

print(f"Top features list saved to {top_features_path}")

# Optional: Upload results to S3
try:
    s3 = boto3.resource('s3')
    bucket_name = 'is459-g1t7-smart-meters-in-london'
    
    # Upload CSV
    s3_csv_key = "feature-importance/feature_importances.csv"
    s3.Bucket(bucket_name).upload_file(csv_path, s3_csv_key)
    
    # Upload plot
    s3_plot_key = "feature-importance/feature_importances_plot.png"
    s3.Bucket(bucket_name).upload_file(plot_path, s3_plot_key)
    
    # Upload model
    s3_model_key = "ml-models/rf_feature_importance.pkl"
    s3.Bucket(bucket_name).upload_file(model_path, s3_model_key)
    
    # Upload top features
    s3_features_key = "feature-importance/top_features.txt"
    s3.Bucket(bucket_name).upload_file(top_features_path, s3_features_key)
    
    print(f"\nResults uploaded to S3 bucket: {bucket_name}")
except Exception as e:
    print(f"Warning: Could not upload to S3: {str(e)}")

# Stop Spark Session
spark.stop()