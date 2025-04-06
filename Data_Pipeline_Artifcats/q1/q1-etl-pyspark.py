from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, to_timestamp
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import os

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessingQ1").getOrCreate()

    # ---------- READ INPUT FILES ----------
    df2 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "halfhourly_dataset"), header=True, inferSchema=True)
    df4 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "acorn_details.csv"), header=True, inferSchema=True)
    df6 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "informations_households.csv"), header=True, inferSchema=True)
    df10_1 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "acorn_information.csv"), header=True, inferSchema=True)
    df10_reduced = df10_1.select("Acorn", "Acorn Category")
    df12 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "tariff_type.csv"), header=True, inferSchema=True)
    df14 = spark.read.csv(os.path.join(S3_INPUT_FOLDER, "property_type_energy_efficiency.csv"), header=True, inferSchema=True)
    
    # ---------- CAST COLUMNS ----------

    df2 = df2.withColumn("LCLid", col("LCLid").cast("string"))
    df2 = df2.withColumn("tstp", col("tstp").cast("string"))
    df2 = df2.withColumn("energy(kWh/hh)", col("energy(kWh/hh)").cast("double"))

    df4 = df4.withColumn("MAIN CATEGORIES", col("MAIN CATEGORIES").cast("string"))
    df4 = df4.withColumn("CATEGORIES", col("CATEGORIES").cast("string"))
    df4 = df4.withColumn("REFERENCE", col("REFERENCE").cast("string"))

    df6 = df6.withColumn("stdorToU", col("stdorToU").cast("string"))
    df6 = df6.withColumn("Acorn", col("Acorn").cast("string"))
    df6 = df6.withColumn("Acorn_grouped", col("Acorn_grouped").cast("string"))

    df10_reduced = df10_reduced.withColumn("Acorn", col("Acorn").cast("string"))
    df10_reduced = df10_reduced.withColumn("Acorn Category", col("Acorn Category").cast("string"))

    df12 = df12.withColumn("TariffDateTime", col("TariffDateTime").cast("string"))
    df12 = df12.withColumn("Tariff", col("Tariff").cast("string"))
    df12 = df12.withColumn("TariffDateTime", to_timestamp("TariffDateTime", "M/d/yy HH:mm"))

    df14 = df14.withColumn("Housing Type", col("Housing Type").cast("string"))
    df14 = df14.withColumn("Current Efficiency", col("Current Efficiency").cast("double"))
    df14 = df14.withColumn("Potential Efficiency", col("Potential Efficiency").cast("double"))
    df14 = df14.withColumn("Difference", col("Difference").cast("double"))

    print("\n----------------------------------------------------")
    print("Reading input files:")
    print(f"Input df2 columns: {df2.columns}")
    df2.show(5)

    print(f"Input df4 columns: {df4.columns}")
    df4.show(5)
    
    print(f"Input df6 columns: {df6.columns}")
    df6.show(5)
    
    print(f"Input df10 columns: {df10_reduced.columns}")
    df10_reduced.show(5)
    
    print(f"Input df12 columns: {df12.columns}")
    df12.show(5)

    print(f"Input df14 columns: {df14.columns}")
    df14.show(5)

    # ---------- DROP UNNECESSARY ROWS ----------

    df2 = df2.na.drop(subset=["energy(kWh/hh)"])
    df6 = df6.drop(df6["file"])

    # ---------- PROCESS ACORN DETAILS ----------
    # Emulate melt with stack:
    # Adjust the number of rows (N) in the stack() function based on the actual number of ACORN columns
    print(df4.columns)
    df4_melt = df4.selectExpr(
        "`MAIN CATEGORIES`",
        "CATEGORIES",
        "REFERENCE",
        "stack(19, " +
        "  'ACORN-A', cast(`ACORN-A` as double), " +
        "  'ACORN-B', cast(`ACORN-B` as double), " +
        "  'ACORN-C', cast(`ACORN-C` as double), " +
        "  'ACORN-D', cast(`ACORN-D` as double), " +
        "  'ACORN-E', cast(`ACORN-E` as double), " +
        "  'ACORN-F', cast(`ACORN-F` as double), " +
        "  'ACORN-G', cast(`ACORN-G` as double), " +
        "  'ACORN-H', cast(`ACORN-H` as double), " +
        "  'ACORN-I', cast(`ACORN-I` as double), " +
        "  'ACORN-J', cast(`ACORN-J` as double), " +
        "  'ACORN-K', cast(`ACORN-K` as double), " +
        "  'ACORN-L', cast(`ACORN-L` as double), " +
        "  'ACORN-M', cast(`ACORN-M` as double), " +
        "  'ACORN-N', cast(`ACORN-N` as double), " +
        "  'ACORN-O', cast(`ACORN-O` as double), " +
        "  'ACORN-P', cast(`ACORN-P` as double), " +
        "  'ACORN-Q', cast(`ACORN-Q` as double) " +
        ") as (Acorn, Value)"
    )

    # Cast column types
    df4_melt = df4_melt.withColumn("Acorn", col("Acorn").cast("string"))
    df4_melt = df4_melt.withColumn("Value", col("Value").cast("double"))

    # Calculate Total and Percentage using a window partitioned by MAIN CATEGORIES, CATEGORIES, and Acorn
    window_spec = Window.partitionBy("MAIN CATEGORIES", "CATEGORIES", "Acorn")
    df4_melt = df4_melt.withColumn("Total", F.sum("Value").over(window_spec))
    df4_melt = df4_melt.withColumn("Percentage", col("Value") / col("Total"))

    # Cast column types
    df4_melt = df4_melt.withColumn("Percentage", col("Percentage").cast("double"))

    # Filter for HOUSING and House Type rows
    df4_house_filtered = df4_melt.filter((col("MAIN CATEGORIES") == "HOUSING") & (col("CATEGORIES") == "House Type"))

    # ---------- MERGE DATAFRAMES ----------

    # MERGE 1: df4 & df14 --> df4_df14
    print("\n----------------------------------------------------")
    print("Merging df4 & df14")
    merged_df4_df14 = df4_house_filtered.join(
        df14,
        df4_house_filtered["REFERENCE"] == df14["Housing Type"],
        how="left"
    ).drop("REFERENCE", "MAIN CATEGORIES", "CATEGORIES", "Value", "Housing Type")

    # Calculate Efficiency and Potential Values
    merged_df4_df14 = merged_df4_df14.withColumn("Efficiency_Value", col("Current Efficiency") * col("Percentage")) \
                                     .withColumn("Potential_Value", col("Potential Efficiency") * col("Percentage"))

    # Group by Acorn and sum the values
    final_df4_df14 = merged_df4_df14.groupBy("Acorn").agg(
        F.sum("Efficiency_Value").alias("Efficiency_Value"),
        F.sum("Potential_Value").alias("Potential_Value")
    )
    final_df4_df14 = final_df4_df14.withColumn("Difference_Value", col("Potential_Value") - col("Efficiency_Value"))

    # Drop unnecessary columns
    final_df4_df14 = final_df4_df14.drop('Current Efficiency', 'Potential Efficiency')

    # Cast column types
    final_df4_df14 = final_df4_df14.withColumn("Efficiency_Value", col("Efficiency_Value").cast("double"))
    final_df4_df14 = final_df4_df14.withColumn("Potential_Value", col("Potential_Value").cast("double"))
    final_df4_df14 = final_df4_df14.withColumn("Difference_Value", col("Difference_Value").cast("double"))

    print("After merging df4 and df14:")
    print(f"Columns: {final_df4_df14.columns}")
    final_df4_df14.show(10)

    # MERGE 2: df2 & df6 --> df2_df6
    print("\n----------------------------------------------------")
    print("Merging df2 & df6:")
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left")
    print("After merging df2 and df6:")
    print(f"Columns: {merged_df2_df6.columns}")
    merged_df2_df6.show(10)

    # MERGE 3: df2_df6 & df10 --> df2_df6_df10
    print("\n----------------------------------------------------")
    print("Merging df2 & df6 & df10:")
    merged_df2_df6_df10 = merged_df2_df6.join(df10_reduced, on="Acorn", how="left")
    print("After merging with acorn info (df10):")
    print(f"Columns: {merged_df2_df6_df10.columns}")
    merged_df2_df6_df10.show(10)
    
    # MERGE 4: df2_df6_df10 & df12 --> df2_df6_df10_df12
    print("\n----------------------------------------------------")
    print("Merging df2 & df6 & df10 & df12:")
    merged_df2_df6_df10_df12 = merged_df2_df6_df10.join(df12, merged_df2_df6_df10["tstp"] == df12["TariffDateTime"], how="inner")
    merged_df2_df6_df10_df12 = merged_df2_df6_df10_df12.drop(df12["TariffDateTime"])
    print("After merging with tariff info (df12):")
    print(f"Columns: {merged_df2_df6_df10_df12.columns}")
    merged_df2_df6_df10_df12.show(10)

    # MERGE 5: df2_df6_df10_df12 & df4_df14 --> df2_df4_df6_df10_df12_df14
    print("\n----------------------------------------------------")
    print("Merging df2 & df6 & df10 & df12 & df4 & df14:")
    merged_df2_df4_df6_df10_df12_df14 = merged_df2_df6_df10_df12.join(final_df4_df14, on="Acorn", how="left")
    print("After merging with household demographics info (df4_df14):")
    print(f"Columns: {merged_df2_df4_df6_df10_df12_df14.columns}")
    merged_df2_df4_df6_df10_df12_df14.show(10)

    print("\n----------------------------------------------------")
    print("Schema of final DataFrame:")
    merged_df2_df4_df6_df10_df12_df14.printSchema()

    print(f"Number of rows: {merged_df2_df4_df6_df10_df12_df14.count()}")
    merged_df2_df4_df6_df10_df12_df14.show(10)

    # ---------- WRITE THE FINAL DATAFRAMES TO S3 AS PARQUET ----------

    # df4_melt -> final_q1_df/df4
    df4_melt.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}final_q1_df/df4_melt")
    print("df4_melt successfully written to S3 as Parquet in folder final_q1_df | subfolder df4_melt!")

    # merged_df2_df4_df6_df10_df12_df14 -> final_q1_df/merged_df2_df4_df6_df10_df12_df14
    merged_df2_df4_df6_df10_df12_df14.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}final_q1_df/merged_df2_df4_df6_df10_df12_df14")
    print("merged_df2_df4_df6_df10_df12_df14 successfully written to S3 as Parquet in folder final_q1_df | subfolder merged_df2_df4_df6_df10_df12_df14!")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()