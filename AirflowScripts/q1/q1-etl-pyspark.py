from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, mean

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessingQ1").getOrCreate()

    # ---------- READ INPUT FILES ----------

    # Read CSV files
    df2 = spark.read.csv(f"{S3_INPUT_FOLDER}halfhourly_dataset.csv", header=True, inferSchema=True)
    df6 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
    df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
    df10_reduced = df10_1.select("Acorn", "Acorn Category")     # Reduce df10_1 to required columns
    df12 = spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .load(f"{S3_INPUT_FOLDER}tariff_type.xlsx")

    # ---------- CONVERT COLUMN TYPES ----------

    df2 = df2.withColumn("tstp", to_timestamp(col("tstp")))
    df12 = df12.withColumn("TariffDateTime", to_timestamp(col("TariffDateTime")))
    df2 = df2.withColumn("energy(kWh/hh)", col("energy(kWh/hh)").cast("double"))

    # ---------- DROP UNNECESSARY ROWS ----------

    # Drop any rows where energy conversion failed
    df2 = df2.na.drop(subset=["energy(kWh/hh)"])

    # Filter df2 to only include rows with a matching TariffDateTime in df12
    # Using an inner join to keep only matching timestamps
    tariff_dt = df12.select("TariffDateTime").distinct()
    df2 = df2.join(tariff_dt, df2.tstp == tariff_dt.TariffDateTime, "inner")

    # ---------- MERGE DATAFRAMES ----------

    # Merge #1: df2 & df6
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left").drop("file")

    # Merge #2: with df10
    merged_df2_df6_df10 = merged_df2_df6.join(df10_reduced, on="Acorn", how="left")

    # Merge #3: with df12
    acorn_energy = merged_df2_df6_df10.groupBy("tstp", "Acorn", "Acorn_grouped", "Acorn Category") \
                                    .agg(mean("energy(kWh/hh)").alias("mean_energy"))
    merged_df2_df6_df10_df12 = df12.join(acorn_energy, df12.TariffDateTime == acorn_energy.tstp, "left")

    # ---------- WRITE THE FINAL DATAFRAME TO S3 AS PARQUET ----------

    merged_df2_df6_df10_df12.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}final_q1_df")
    print("Processed datasets successfully written to S3 as Parquet!")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
