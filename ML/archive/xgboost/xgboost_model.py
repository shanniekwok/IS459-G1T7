import pyspark
import findspark
import os
import shutil
import glob

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.sql.functions import col, dayofweek, when, month, hour, lag, avg, stddev, sum as spark_sum, expr
from pyspark.sql.window import Window
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder

findspark.init()

# Create Spark session
spark = SparkSession.builder.appName("EnergyPrediction").config("spark.driver.memory", "12g").config("spark.executor.memory", "6g").getOrCreate()

# Load Parquet file
repo_root = os.getcwd()  
parquet_path = os.path.join(repo_root, "merged_df1_df3_df7_df8")
df = spark.read.parquet(parquet_path)
df.show(5)
df.printSchema()

# Time-Based Features
df = df.withColumn("day_of_week", dayofweek(col("day")))
df = df.withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))
df = df.withColumn("month", month(col("day")))
df = df.withColumn("hour", hour(col("time")))

# # Lag Features
# window_spec = Window.partitionBy("LCLid").orderBy("day")
# df = df.withColumn("lag_1_day", lag("energy_sum", 1).over(window_spec))
# df = df.withColumn("lag_2_day", lag("energy_sum", 2).over(window_spec))
# df = df.withColumn("lag_7_day", lag("energy_sum", 7).over(window_spec))

# # Rolling Aggregates
# rolling_window_3d = window_spec.rowsBetween(-2, 0)
# rolling_window_7d = window_spec.rowsBetween(-6, 0)
# rolling_window_14d = window_spec.rowsBetween(-13, 0)
# rolling_window_30d = window_spec.rowsBetween(-29, 0)

# df = df.withColumn("rolling_energy_3d", spark_sum("energy_sum").over(rolling_window_3d))
# df = df.withColumn("rolling_energy_7d", spark_sum("energy_sum").over(rolling_window_7d))
# df = df.withColumn("rolling_energy_14d", spark_sum("energy_sum").over(rolling_window_14d))
# df = df.withColumn("rolling_energy_30d", spark_sum("energy_sum").over(rolling_window_30d))

# df = df.withColumn("rolling_energy_mean_14d", avg("energy_sum").over(rolling_window_14d))
# df = df.withColumn("rolling_energy_std_14d", stddev("energy_sum").over(rolling_window_14d))

# # Energy Consumption Trends
# df = df.withColumn("energy_trend_3d", col("rolling_energy_3d") - col("lag_1_day"))
# df = df.withColumn("energy_trend_7d", col("rolling_energy_7d") - col("lag_1_day"))

# # Daily Energy Change Percentage
# df = df.withColumn("daily_energy_change", (col("lag_1_day") - col("energy_sum")) / (col("lag_1_day") + 1))

# Weather Interactions
df = df.withColumn("temperature_variability", col("temperatureMax") - col("temperatureMin"))
df = df.withColumn("humidity_temp_interaction", col("humidity") * col("temperatureMax"))
df = df.withColumn("cloud_temp_interaction", col("cloudCover") * col("temperatureMax"))

# Fill Missing Values
df = df.dropna(subset=["energy_sum"])  

df = df.fillna({
    # "lag_1_day": 0, "lag_2_day": 0, "lag_7_day": 0,
    # "rolling_energy_3d": df.select(avg("energy_sum")).collect()[0][0],
    # "rolling_energy_7d": df.select(avg("energy_sum")).collect()[0][0],
    # "rolling_energy_14d": df.select(avg("energy_sum")).collect()[0][0],
    # "rolling_energy_30d": df.select(avg("energy_sum")).collect()[0][0],
    # "rolling_energy_mean_14d": df.select(avg("energy_sum")).collect()[0][0],
    # "rolling_energy_std_14d": df.select(stddev("energy_sum")).collect()[0][0],
    # "energy_trend_3d": 0, "energy_trend_7d": 0,
    # "daily_energy_change": 0,
    "temperature_variability": df.select(avg("temperatureMax") - avg("temperatureMin")).collect()[0][0],
    "humidity_temp_interaction": df.select(avg("humidity") * avg("temperatureMax")).collect()[0][0],
    "cloud_temp_interaction": df.select(avg("cloudCover") * avg("temperatureMax")).collect()[0][0],
})

# Convert 'Type' into a binary holiday indicator
df = df.withColumn("is_holiday", when(col("Type").isNotNull(), 1).otherwise(0))

# Drop original "Type" column after conversion
df = df.drop("Type")

# Feature Engineering
feature_cols = [
    "day_of_week", "is_weekend", "month", "hour",
    # "lag_1_day", "lag_2_day", "lag_7_day",
    # "rolling_energy_3d", "rolling_energy_7d", "rolling_energy_14d", "rolling_energy_30d",
    # "rolling_energy_mean_14d", "rolling_energy_std_14d",
    # "energy_trend_3d", "energy_trend_7d",
    # "daily_energy_change",
    "temperature_variability", "humidity_temp_interaction",
    "cloud_temp_interaction"
]

# Vector Assembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df = assembler.transform(df)

# Feature Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
df = scaler.fit(df).transform(df)

# Train-Test Split
train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

# Optimized XGBoost Model
xgb = SparkXGBRegressor(
    features_col="scaled_features",
    label_col="energy_sum",
    max_depth=5,  
    eta=0.1,  
    subsample=0.85,  
    colsample_bytree=0.85,  
    min_child_weight=5,  
    alpha=0.5,  # L1 regularization
    n_estimators=600,
    # lambda_=1.5,  # L2 regularization
    # num_round=600,  
)

xgb_model = xgb.fit(train_df)

# # Save the trained model
# xgb_model_path = os.path.join(repo_root, "xgboost_model")
# xgb_model.save(xgb_model_path) 

predictions = xgb_model.transform(test_df)
predictions = predictions.select("day", "LCLid", "is_holiday", "energy_sum", "prediction")

# # Save predictions in CSV
# repo_root = os.getcwd()  
# output_folder = os.path.join(repo_root, "xgboost_output")  
# os.makedirs(output_folder, exist_ok=True)
# temp_output_folder = os.path.join(repo_root, "temp_output")
# predictions.repartition(1).write.mode("overwrite").csv(temp_output_folder, header=True)
# part_file = glob.glob(os.path.join(temp_output_folder, "part-*.csv"))[0]
# final_output_path = os.path.join(output_folder, "xgboost_predictions.csv")
# shutil.move(part_file, final_output_path)
# shutil.rmtree(temp_output_folder)

# Evaluate Model
evaluator = RegressionEvaluator(labelCol="energy_sum", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Test RMSE (XGBoost): {rmse}")

# mean_energy_sum = df.select(avg("energy_sum")).collect()[0][0]
# rmse_percentage = (rmse / mean_energy_sum) * 100
# print(f"RMSE as percentage of mean: {rmse_percentage:.2f}%")

# Stop Spark Session
spark.stop()