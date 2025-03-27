from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.types import DecimalType

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessingQ1").getOrCreate()

    # ---------- READ INPUT FILES ----------
    df2 = spark.read.csv(f"{S3_INPUT_FOLDER}halfhourly_dataset.csv", header=True, inferSchema=True)
    df6 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
    df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
    df10_reduced = df10_1.select("Acorn", "Acorn Category")
    df12 = spark.read.csv(f"{S3_INPUT_FOLDER}tariff_type.csv", header=True, inferSchema=True)

    # ---------- CAST COLUMNS ----------
    df2 = df2.withColumn("LCLid", col("LCLid").cast("string"))
    df2 = df2.withColumn("tstp", col("tstp").cast("string"))
    df2 = df2.withColumn("energy(kWh/hh)", col("energy(kWh/hh)").cast("decimal"))

    df6 = df6.withColumn("stdorToU", col("stdorToU").cast("string"))
    df6 = df6.withColumn("Acorn", col("Acorn").cast("string"))
    df6 = df6.withColumn("Acorn_grouped", col("Acorn_grouped").cast("string"))

    df10_reduced = df10_reduced.withColumn("Acorn", col("Acorn").cast("string"))
    df10_reduced = df10_reduced.withColumn("Acorn Category", col("Acorn Category").cast("string"))

    df12 = df12.withColumn("TariffDateTime", col("TariffDateTime").cast("string"))
    df12 = df12.withColumn("Tariff", col("Tariff").cast("string"))

    print("\n----------------------------------------------------")
    print("Reading input files:")
    print(f"Input df2 columns: {df2.columns}")
    print(f"Input df6 columns: {df6.columns}")
    print(f"Input df10 columns: {df10_reduced.columns}")
    print(f"Input df12 columns: {df12.columns}")

    # ---------- DROP UNNECESSARY ROWS ----------
    df2 = df2.na.drop(subset=["energy(kWh/hh)"])
    
    # Use an alias for df12 in the first join so that the original df12 remains intact
    df2 = df2.join(df12.alias("df12_first"), df2.tstp == col("df12_first.TariffDateTime"), "inner")

    print("\n----------------------------------------------------")
    print("After merging df2 with df12 (aliased as df12_first):")
    print(f"df2 columns: {df2.columns}")

    # ---------- MERGE DATAFRAMES ----------
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left")
    print("\n----------------------------------------------------")
    print("After merging df2 and df6:")
    print(f"Columns: {merged_df2_df6.columns}")

    merged_df2_df6_df10 = merged_df2_df6.join(df10_reduced, on="Acorn", how="left")
    print("\n----------------------------------------------------")
    print("After merging with acorn info (df10):")
    print(f"Columns: {merged_df2_df6_df10.columns}")

    # ---------- AGGREGATE ENERGY BY ACORN DETAILS ----------
    acorn_energy = merged_df2_df6_df10.groupBy("tstp", "Acorn", "Acorn_grouped", "Acorn Category") \
                                    .agg(mean("energy(kWh/hh)").alias("mean_energy"))

    # ---------- FINAL JOIN ----------
    # Here we use the original df12 for the final join
    merged_final = df12.join(acorn_energy, df12.TariffDateTime == acorn_energy.tstp, "left")
    print("\n----------------------------------------------------")
    print("After final join with df12 and acorn_energy:")
    print(f"Final columns: {merged_final.columns}")

    print("\n----------------------------------------------------")
    print("Schema of final DataFrame:")
    merged_final.printSchema()

    print(f"Number of rows: {merged_final.count()}")
    merged_final.show(5)

    # ---------- WRITE THE FINAL DATAFRAME TO S3 AS PARQUET ----------
    merged_final.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}final_q1_df")
    print("Processed datasets successfully written to S3 as Parquet!")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()