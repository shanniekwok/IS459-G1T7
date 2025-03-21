from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, to_timestamp

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

# ====================================================================================================================================

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessing").getOrCreate()

    # Read input CSV files from S3
    df2 = spark.read.csv(f"{S3_INPUT_FOLDER}halfhourly_dataset.csv", header=True, inferSchema=True)
    df6 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
    df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
    df10_2 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_category_information.csv", header=True, inferSchema=True)
    df12 = spark.read.csv(f"{S3_INPUT_FOLDER}tariff_information.csv", header=True, inferSchema=True)

    # Convert column types
    df2 = df2.withColumn("tstp", to_timestamp(col("tstp")))
    df12 = df12.withColumn("TariffDateTime", to_timestamp(col("TariffDateTime")))
    df2 = df2.withColumn("energy(kWh/hh)", col("energy(kWh/hh)").cast("double"))

    # Drop rows where energy conversion failed
    df2 = df2.na.drop(subset=["energy(kWh/hh)"])

    # Merge df10_1 and df10_2 on "Acorn Category"
    df10 = df10_1.join(df10_2, on="Acorn Category", how="left")

    # Reduce df2 to only rows that have matching datetimes in df12
    df2 = df2.join(df12.select("TariffDateTime").distinct(), df2.tstp == df12.TariffDateTime, "inner")

    # Merge reduced df2 & df6
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left").drop("file")

    # Merge with df10
    merged_df2_df6_df10 = merged_df2_df6.join(df10, on="Acorn", how="left")

    # Compute average energy for each Acorn
    acorn_avg = merged_df2_df6_df10.groupBy("tstp", "Acorn").agg(mean("energy(kWh/hh)").alias("avg_energy"))

    # Pivot so that each Acorn category is a column
    acorn_pivot = acorn_avg.groupBy("tstp").pivot("Acorn").agg(mean("avg_energy"))

    # Compute average energy for each Acorn_grouped
    acorn_grouped_avg = merged_df2_df6_df10.groupBy("tstp", "Acorn_grouped").agg(mean("energy(kWh/hh)").alias("avg_energy"))
    acorn_grouped_pivot = acorn_grouped_avg.groupBy("tstp").pivot("Acorn_grouped").agg(mean("avg_energy"))

    # Compute average energy for each Acorn Category
    acorn_category_avg = merged_df2_df6_df10.groupBy("tstp", "Acorn Category").agg(mean("energy(kWh/hh)").alias("avg_energy"))
    acorn_category_pivot = acorn_category_avg.groupBy("tstp").pivot("Acorn Category").agg(mean("avg_energy"))

    # Merge the aggregated results with df12
    merged_df12_acorn = df12.join(acorn_pivot, df12.TariffDateTime == acorn_pivot.tstp, "left").drop("tstp")
    merged_df12_acorn = merged_df12_acorn.withColumnRenamed("TariffDateTime", "Datetime")

    merged_df12_acorn_grouped = df12.join(acorn_grouped_pivot, df12.TariffDateTime == acorn_grouped_pivot.tstp, "left").drop("tstp")
    merged_df12_acorn_grouped = merged_df12_acorn_grouped.withColumnRenamed("TariffDateTime", "Datetime")

    merged_df12_acorn_category = df12.join(acorn_category_pivot, df12.TariffDateTime == acorn_category_pivot.tstp, "left").drop("tstp")
    merged_df12_acorn_category = merged_df12_acorn_category.withColumnRenamed("TariffDateTime", "Datetime")

    # Write the final merged datasets to S3 in Parquet format
    merged_df12_acorn.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df12_acorn")
    merged_df12_acorn_grouped.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df12_acorn_grouped")
    merged_df12_acorn_category.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df12_acorn_category")

    print("Processed datasets successfully written to S3 as Parquet.")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()

# ====================================================================================================================================

# # [TESTED, WORKS] OLD CODE

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, mean, to_timestamp

# # S3 Paths
# S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
# S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

# def main():
#     # Create Spark Session
#     spark = SparkSession.builder.appName("SmartMetersProcessing").getOrCreate()

#     # Read input CSV files from S3
#     df1 = spark.read.csv(f"{S3_INPUT_FOLDER}daily_dataset.csv", header=True, inferSchema=True)
#     df5 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
#     df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
#     df10_2 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_category_information.csv", header=True, inferSchema=True)

#     # Merge df10_1 and df10_2 on "Acorn Category"
#     df10 = df10_1.join(df10_2, on="Acorn Category", how="left")

#     # Merge df1 and df5 on "LCLid"
#     merged_df1_df5 = df1.join(df5, on="LCLid", how="left")

#     # Merge the above result with df10 on "Acorn"
#     final_merged_df = merged_df1_df5.join(df10, on="Acorn", how="left")

#     # Print the number of records after merging
#     print("Total number of records after merging:", final_merged_df.count())

#     # Write the final merged dataset to S3 in **Parquet format**
#     final_merged_df.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df1_df5_df10")

#     print("Merged dataset successfully written to S3 as Parquet:", S3_OUTPUT_FOLDER)

#     # Stop Spark session
#     spark.stop()

# if __name__ == "__main__":
#     main()