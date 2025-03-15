import pyspark
import findspark
import os
import shutil
import glob

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.functions import col, dayofweek, when, month, hour, lag, avg, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.ml.feature import MinMaxScaler
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator

findspark.init()

# Create Spark session
spark = SparkSession.builder.appName("ParquetViewer").config("spark.driver.memory", "8g").config("spark.executor.memory", "4g").getOrCreate()

df = spark.read.parquet("/Users/caitlinyap/GitHub/IS459-G1T7/ML/merged_df1_df3_df7_df8/")

df.show(5)
df.printSchema()

# Time-based features
df = df.withColumn("day_of_week", dayofweek(col("day")))
df = df.withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))
df = df.withColumn("month", month(col("day")))
df = df.withColumn("hour", hour(col("time")))

# Lag feature (previous day's energy consumption)
window_spec = Window.partitionBy("LCLid").orderBy("day")
df = df.withColumn("lag_1_day", lag("energy_sum", 1).over(window_spec))

# Weather-based features
df = df.withColumn("temperature_variability", col("temperatureMax") - col("temperatureMin"))
df = df.withColumn("humidity_temp_interaction", col("humidity") * col("temperatureMax"))
df = df.withColumn("cloud_temp_interaction", col("cloudCover") * col("temperatureMax"))

# Rolling sum of precipitation over 7 days
df = df.withColumn("rolling_precipitation_7d", spark_sum("precipType").over(window_spec.rowsBetween(-6, 0)))

df = df.dropna(subset=["energy_sum"])
df = df.fillna({
    "lag_1_day": 0,  
    "temperature_variability": df.select(avg("temperatureMax") - avg("temperatureMin")).collect()[0][0],
    "humidity_temp_interaction": df.select(avg("humidity") * avg("temperatureMax")).collect()[0][0],
    "cloud_temp_interaction": df.select(avg("cloudCover") * avg("temperatureMax")).collect()[0][0],
    "rolling_precipitation_7d": 0  
})
columns_to_drop = ["precipType", "summary", "Type"]
df = df.drop(*columns_to_drop)

# Assemble features
feature_cols = [
    "day_of_week", "is_weekend", "lag_1_day",
    "temperature_variability", "humidity_temp_interaction",
    "cloud_temp_interaction", "rolling_precipitation_7d"
]
if "features" in df.columns:
    df = df.drop("features")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
df = scaler.fit(df).transform(df)

# Train-Test Split
train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

# print(f"Training set size: {train_df.count()}")
# print(f"Test set size: {test_df.count()}")

if train_df.count() == 0:
    raise ValueError("Training dataset is empty. Check preprocessing!")

os.environ["DMLC_TRACKER_URI"] = "127.0.0.1"
os.environ["DMLC_TRACKER_PORT"] = "9091"
os.environ["DMLC_NUM_WORKER"] = "1"
os.environ["DMLC_NUM_SERVER"] = "1"

## XGBoost Model
train_df_xgb, test_df_xgb = df.randomSplit([0.9, 0.1], seed=42)

xgb = SparkXGBRegressor(
    features_col="scaled_features",
    label_col="energy_sum",
    max_depth=10,
    eta=0.05,
    subsample=0.8,
    # num_round=150,
    # n_workers=1,  # Run in single-worker mode
    # use_external_storage=False
)

xgb_model = xgb.fit(train_df_xgb)

predictions = xgb_model.transform(test_df_xgb)
predictions = predictions.orderBy("day", "LCLid")

# save predictions in csv
repo_root = os.getcwd()  
output_folder = os.path.join(repo_root, "xgboost_output")  
os.makedirs(output_folder, exist_ok=True)
temp_output_folder = os.path.join(repo_root, "temp_output")
predictions.select("day", "LCLid", "energy_sum", "prediction").repartition(1).write.mode("overwrite").csv(temp_output_folder, header=True)
part_file = glob.glob(os.path.join(temp_output_folder, "part-*.csv"))[0]
final_output_path = os.path.join(output_folder, "xgboost_predictions.csv")
shutil.move(part_file, final_output_path)
shutil.rmtree(temp_output_folder)

evaluator = RegressionEvaluator(labelCol="energy_sum", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Test RMSE (XGBoost): {rmse}")
mean_energy_sum = df.select(avg("energy_sum")).collect()[0][0]
rmse_percentage = (4.62 / mean_energy_sum) * 100
print(f"RMSE as percentage of mean: {rmse_percentage:.2f}%")