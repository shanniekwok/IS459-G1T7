# Datasets needed:
# df1 daily_dataset.csv
# df2 halfhourly_dataset.csv (greater granularity)
# df3 hhblock_dataset.zip (block files of transposed data of a household's energy 
# consumption in a day; hh_0 column = consumption from 00:00 - 00:30, and so on)
# df7 uk_bank_holidays.csv 
# df8 weather_daily_darksky.csv
# df9 weather_hourly_darksky.csv (greater granularity)

# Output: merged_q2_daily (Parquet format) -> df1 + df8 + df7
#         merged_q2_hourly (Parquet format) -> df2 + df9 + df7

# import sys
# from datetime import datetime

from pyspark.sql import SparkSession
# from pyspark.sql.functions import (
#     col, when, to_date, date_format,
#     dayofweek, month, hour, lag, avg,
#     sum as spark_sum
# )
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession\
        .builder\
        .appName("IS459-Project-Q2")\
        .getOrCreate()

    # Read existing CSV datasets from S3
    df1 = spark.read.csv(f"{S3_INPUT_FOLDER}daily_dataset.csv", header=True, inferSchema=True)
    df2 = spark.read.csv(f"{S3_INPUT_FOLDER}halfhourly_dataset.csv", header=True, inferSchema=True)
    df7 = spark.read.csv(f"{S3_INPUT_FOLDER}uk_bank_holidays.csv", header=True, inferSchema=True)
    df8 = spark.read.csv(f"{S3_INPUT_FOLDER}weather_daily_darksky.csv", header=True, inferSchema=True)
    df9 = spark.read.csv(f"{S3_INPUT_FOLDER}weather_hourly_darksky.csv", header=True, inferSchema=True)

    # Read existing zip datasets from S3
    # hhblock_path = "s3://is459-g1t7-smart-meters-in-london/raw-data/hhblock_dataset/block_*.csv"
    # df3 = spark.read.csv(hhblock_path, header=True, inferSchema=True)

    # Rename df1's `day` column to `date` and convert to datetime format
    df1 = df1.withColumnRenamed("day", "date").withColumn("date", to_date(col("date"), "dd/MM/yyyy"))

    # Create new columns for time-based features
    df1 = (df1.withColumn("day_of_week", dayofweek(col("date")))
            .withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))
            .withColumn("month", month(col("date"))))

    # Lag feature (previous day's energy consumption)
    window_spec = Window.partitionBy("LCLid").orderBy("date")
    df1 = df1.withColumn("prev_day_energy_sum", lag("energy_sum", 1).over(window_spec))

    # Drop rows where `energy_sum` is NULL
    df1 = df1.dropna(subset=["energy_sum"])

    # Fill missing values instead of dropping rows in df1
    df1 = df1.fillna({
        "prev_day_energy_sum": 0
    })

    # Convert df8's `time` column to `date` and extract hour
    df8 = (df8.withColumn("date", to_date(col("time"), "dd/MM/yyyy HH:mm"))
            .withColumn("hour", col("time").substr(12, 2).cast("int")))

    # Create new columns for weather-based features
    df8 = df8.withColumn("temperature_variability", col("temperatureMax") - col("temperatureMin"))
    df8 = df8.withColumn("humidity_temp_interaction", col("humidity") * col("temperatureMax"))
    df8 = df8.withColumn("cloud_temp_interaction", col("cloudCover") * col("temperatureMax"))

    # Calculate average values for the new weather-based features
    avg_temp_variability = (df8.select((avg(col("temperatureMax")) - avg(col("temperatureMin"))).alias("avg_temp_variability"))
                                .collect()[0][0])
    avg_humidity_temp_interaction = (df8.select((avg(col("humidity")) * avg(col("temperatureMax"))).alias("avg_humidity_temp_interaction"))
                                    .collect()[0][0])
    avg_cloud_temp_interaction = (df8.select((avg(col("cloudCover")) * avg(col("temperatureMax"))).alias("avg_cloud_temp_interaction"))
                                    .collect()[0][0])

    # Rolling sum of precipitation over 7 days
    # df8 = df8.withColumn("rolling_precipitation_7d", spark_sum("precipType").over(window_spec.rowsBetween(-6, 0)))

    # Fill missing values instead of dropping rows in df8
    df8 = df8.fillna({
        "temperature_variability": avg_temp_variability,
        "humidity_temp_interaction": avg_humidity_temp_interaction,
        "cloud_temp_interaction": avg_cloud_temp_interaction,
        # "rolling_precipitation_7d": 0
    })

    # Merge df1 and df8 on `date` column
    merged_df1_df8 = df1.join(df8, on="date", how="left")

    # Rename df7's `Bank Holidays` column to `date` and convert to datetime format
    df7 = (df7.withColumn("Bank holidays", to_date(col("Bank holidays"), "yyyy-MM-dd"))
            .withColumnRenamed("Bank holidays", "date")
            .withColumnRenamed("Type", "holiday"))
    
    # Merge df7 with the previously merged dataframe and add `holiday` column if days overlap
    merged_daily_df1_df7_df8 = merged_df1_df8.join(
        df7.select("date", "holiday"), 
        on="date", 
        how="left"
    )

    # Drop columns with too many missing values
    columns_to_drop = ["energy_std", "uvIndexTime", "moonPhase"]
    for col_name in columns_to_drop:
        if col_name in merged_daily_df1_df7_df8.columns:
            merged_daily_df1_df7_df8 = merged_daily_df1_df7_df8.drop(col_name)
    
    # Check number of records after merging
    print("Total number of records after merging:", merged_daily_df1_df7_df8.count())

    # Write the final merged dataset to S3 in Parquet format
    merged_daily_df1_df7_df8.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_q2_daily")

    print("Merged dataset successfully written to S3 as Parquet:", S3_OUTPUT_FOLDER)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()