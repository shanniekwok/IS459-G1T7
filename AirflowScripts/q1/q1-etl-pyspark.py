from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, to_timestamp
# from pyspark.sql.types import DecimalType

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessingQ1").getOrCreate()

    # ---------- READ INPUT FILES ----------
    df2 = spark.read.csv(f"{S3_INPUT_FOLDER}halfhourly_dataset.csv", header=True, inferSchema=True)
    df6 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
    df6 = df6.drop(df6["file"])
    df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
    df10_reduced = df10_1.select("Acorn", "Acorn Category")
    df12 = spark.read.csv(f"{S3_INPUT_FOLDER}tariff_type.csv", header=True, inferSchema=True)

    # ---------- CAST COLUMNS ----------
    df2 = df2.withColumn("LCLid", col("LCLid").cast("string"))
    df2 = df2.withColumn("tstp", col("tstp").cast("string"))
    df2 = df2.withColumn("energy(kWh/hh)", col("energy(kWh/hh)").cast("double"))

    df6 = df6.withColumn("stdorToU", col("stdorToU").cast("string"))
    df6 = df6.withColumn("Acorn", col("Acorn").cast("string"))
    df6 = df6.withColumn("Acorn_grouped", col("Acorn_grouped").cast("string"))

    df10_reduced = df10_reduced.withColumn("Acorn", col("Acorn").cast("string"))
    df10_reduced = df10_reduced.withColumn("Acorn Category", col("Acorn Category").cast("string"))

    df12 = df12.withColumn("TariffDateTime", col("TariffDateTime").cast("string"))
    df12 = df12.withColumn("Tariff", col("Tariff").cast("string"))
    df12 = df12.withColumn("TariffDateTime", to_timestamp("TariffDateTime", "M/d/yy HH:mm"))

    print("\n----------------------------------------------------")
    print("Reading input files:")
    print(f"Input df2 columns: {df2.columns}")
    df2.show(5)
    
    print(f"Input df6 columns: {df6.columns}")
    df6.show(5)
    
    print(f"Input df10 columns: {df10_reduced.columns}")
    df10_reduced.show(5)
    
    print(f"Input df12 columns: {df12.columns}")
    df12.show(5)

    # ---------- DROP UNNECESSARY ROWS ----------
    df2 = df2.na.drop(subset=["energy(kWh/hh)"])

    # ---------- MERGE DATAFRAMES ----------
    
    # MERGE 1: df2 & df6
    print("\n----------------------------------------------------")
    print("Merging df2 & df6:")
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left")
    print("After merging df2 and df6:")
    print(f"Columns: {merged_df2_df6.columns}")
    merged_df2_df6.show(10)

    # MERGE 2: df2 & df6 & df10
    print("\n----------------------------------------------------")
    print("Merging df2 & df6 & df10:")
    merged_df2_df6_df10 = merged_df2_df6.join(df10_reduced, on="Acorn", how="left")
    print("After merging with acorn info (df10):")
    print(f"Columns: {merged_df2_df6_df10.columns}")
    merged_df2_df6_df10.show(10)
    
    # MERGE 3: df2 & df6 & df10 & df12
    print("\n----------------------------------------------------")
    print("Merging df2 & df6 & df10 & df12:")
    merged_df2_df6_df10_df12 = merged_df2_df6_df10.join(df12, merged_df2_df6_df10["tstp"] == df12["TariffDateTime"], how="inner")
    merged_df2_df6_df10_df12 = merged_df2_df6_df10_df12.drop(df12["TariffDateTime"])
    print("After merging with tariff info (df12):")
    print(f"Columns: {merged_df2_df6_df10_df12.columns}")
    merged_df2_df6_df10_df12.show(10)

    print("\n----------------------------------------------------")
    print("Schema of final DataFrame:")
    merged_df2_df6_df10_df12.printSchema()

    print(f"Number of rows: {merged_df2_df6_df10_df12.count()}")
    merged_df2_df6_df10_df12.show(10)

    # ---------- WRITE THE FINAL DATAFRAME TO S3 AS PARQUET ----------
    merged_df2_df6_df10_df12.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}final_q1_df")
    print("Processed datasets successfully written to S3 as Parquet!")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()