from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, to_timestamp
import os

# Local folder paths (Replace with your actual folder path)
LOCAL_INPUT_FOLDER = "./raw/"
LOCAL_OUTPUT_FOLDER = "./processed/"

def main():
    # Create Spark Session (Enable Local Mode)
    spark = SparkSession.builder.master("local[*]").appName("LocalSmartMetersProcessing").getOrCreate()

    # ---------- READ INPUT FILES LOCALLY ----------
    df2 = spark.read.csv("C:\\Users\\VIGHNESH\\Downloads\\halfhourly_dataset.csv", header=True, inferSchema=True)
    df6 = spark.read.csv(os.path.join(LOCAL_INPUT_FOLDER, "informations_households.csv"), header=True, inferSchema=True)
    df10_1 = spark.read.csv(os.path.join(LOCAL_INPUT_FOLDER, "acorn_information.csv"), header=True, inferSchema=True)
    df10_reduced = df10_1.select("Acorn", "Acorn Category")
    df12 = spark.read.csv(os.path.join(LOCAL_INPUT_FOLDER, "tariff_type.csv"), header=True, inferSchema=True)

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

    # Use an alias for df12 in the first join so that the original df12 remains intact
    df2 = df2.join(df12.alias("df12_first"), df2.tstp == col("df12_first.TariffDateTime"), "inner")

    print("\n----------------------------------------------------")
    print("After merging df2 with df12 (aliased as df12_first):")
    print(f"df2 columns: {df2.columns}")
    df2.show(10)

    # ---------- MERGE DATAFRAMES ----------
    merged_df2_df6 = df2.join(df6, on="LCLid", how="left")
    print("\n----------------------------------------------------")
    print("After merging df2 and df6:")
    print(f"Columns: {merged_df2_df6.columns}")
    merged_df2_df6.show(10)

    merged_df2_df6_df10 = merged_df2_df6.join(df10_reduced, on="Acorn", how="left")
    print("\n----------------------------------------------------")
    print("After merging with acorn info (df10):")
    print(f"Columns: {merged_df2_df6_df10.columns}")
    merged_df2_df6_df10.show(10)

    # ---------- AGGREGATE ENERGY BY ACORN DETAILS ----------
    acorn_energy = merged_df2_df6_df10.groupBy("tstp", "LCLid", "Acorn", "Acorn_grouped", "Acorn Category", "stdorToU") \
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

    # ---------- WRITE THE FINAL DATAFRAME LOCALLY AS PARQUET ----------
    # merged_final.write.mode("overwrite").parquet(os.path.join(LOCAL_OUTPUT_FOLDER, "final_q1_df"))
    # print("Processed datasets successfully written to local folder as Parquet!")
    # merged_final.write.mode("overwrite").option("header", "true").csv(f"{LOCAL_OUTPUT_FOLDER}final_q1_df")
    # print("Processed datasets successfully written to local folder as CSV!")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()


'''
----------------------------------------------------
Reading input files:
Input df2 columns: ['LCLid', 'tstp', 'energy(kWh/hh)']
+---------+-------------------+--------------+
|    LCLid|               tstp|energy(kWh/hh)|
+---------+-------------------+--------------+
|MAC000002|2012-10-12 00:30:00|           0.0|
|MAC000002|2012-10-12 01:00:00|           0.0|
|MAC000002|2012-10-12 01:30:00|           0.0|
|MAC000002|2012-10-12 02:00:00|           0.0|
|MAC000002|2012-10-12 02:30:00|           0.0|
+---------+-------------------+--------------+
only showing top 5 rows

Input df6 columns: ['LCLid', 'stdorToU', 'Acorn', 'Acorn_grouped', 'file']
+---------+--------+-------+-------------+-------+
|    LCLid|stdorToU|  Acorn|Acorn_grouped|   file|
+---------+--------+-------+-------------+-------+
|MAC005492|     ToU| ACORN-|       ACORN-|block_0|
|MAC001074|     ToU| ACORN-|       ACORN-|block_0|
|MAC000002|     Std|ACORN-A|     Affluent|block_0|
|MAC003613|     Std|ACORN-A|     Affluent|block_0|
|MAC003597|     Std|ACORN-A|     Affluent|block_0|
+---------+--------+-------+-------------+-------+
only showing top 5 rows

Input df10 columns: ['Acorn', 'Acorn Category']
+-------+--------------------+
|  Acorn|      Acorn Category|
+-------+--------------------+
|ACORN-A|   Luxury Lifestyles|
|ACORN-B|   Luxury Lifestyles|
|ACORN-C|   Luxury Lifestyles|
|ACORN-D|Established Afflu...|
|ACORN-E|Established Afflu...|
+-------+--------------------+
only showing top 5 rows

Input df12 columns: ['TariffDateTime', 'Tariff']
+-------------------+------+
|     TariffDateTime|Tariff|
+-------------------+------+
|2013-01-01 00:00:00|Normal|
|2013-01-01 00:30:00|Normal|
|2013-01-01 01:00:00|Normal|
|2013-01-01 01:30:00|Normal|
|2013-01-01 02:00:00|Normal|
+-------------------+------+
only showing top 5 rows


----------------------------------------------------
After merging df2 with df12 (aliased as df12_first):
df2 columns: ['LCLid', 'tstp', 'energy(kWh/hh)', 'TariffDateTime', 'Tariff']
+---------+-------------------+--------------+-------------------+------+
|    LCLid|               tstp|energy(kWh/hh)|     TariffDateTime|Tariff|
+---------+-------------------+--------------+-------------------+------+
|MAC000002|2013-01-01 00:00:00|         0.219|2013-01-01 00:00:00|Normal|
|MAC000002|2013-01-01 00:30:00|         0.241|2013-01-01 00:30:00|Normal|
|MAC000002|2013-01-01 01:00:00|         0.191|2013-01-01 01:00:00|Normal|
|MAC000002|2013-01-01 01:30:00|         0.235|2013-01-01 01:30:00|Normal|
|MAC000002|2013-01-01 02:00:00|         0.182|2013-01-01 02:00:00|Normal|
|MAC000002|2013-01-01 02:30:00|         0.229|2013-01-01 02:30:00|Normal|
|MAC000002|2013-01-01 03:00:00|         0.194|2013-01-01 03:00:00|Normal|
|MAC000002|2013-01-01 03:30:00|         0.201|2013-01-01 03:30:00|Normal|
|MAC000002|2013-01-01 04:00:00|         0.122|2013-01-01 04:00:00|Normal|
|MAC000002|2013-01-01 04:30:00|         0.099|2013-01-01 04:30:00|Normal|
+---------+-------------------+--------------+-------------------+------+
only showing top 10 rows


----------------------------------------------------
After merging df2 and df6:
Columns: ['LCLid', 'tstp', 'energy(kWh/hh)', 'TariffDateTime', 'Tariff', 'stdorToU', 'Acorn', 'Acorn_grouped', 'file']
+---------+-------------------+--------------+-------------------+------+--------+-------+-------------+-------+
|    LCLid|               tstp|energy(kWh/hh)|     TariffDateTime|Tariff|stdorToU|  Acorn|Acorn_grouped|   file|
+---------+-------------------+--------------+-------------------+------+--------+-------+-------------+-------+
|MAC000002|2013-01-01 00:00:00|         0.219|2013-01-01 00:00:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 00:30:00|         0.241|2013-01-01 00:30:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 01:00:00|         0.191|2013-01-01 01:00:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 01:30:00|         0.235|2013-01-01 01:30:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 02:00:00|         0.182|2013-01-01 02:00:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 02:30:00|         0.229|2013-01-01 02:30:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 03:00:00|         0.194|2013-01-01 03:00:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 03:30:00|         0.201|2013-01-01 03:30:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 04:00:00|         0.122|2013-01-01 04:00:00|Normal|     Std|ACORN-A|     Affluent|block_0|
|MAC000002|2013-01-01 04:30:00|         0.099|2013-01-01 04:30:00|Normal|     Std|ACORN-A|     Affluent|block_0|
+---------+-------------------+--------------+-------------------+------+--------+-------+-------------+-------+
only showing top 10 rows


----------------------------------------------------
After merging with acorn info (df10):
Columns: ['Acorn', 'LCLid', 'tstp', 'energy(kWh/hh)', 'TariffDateTime', 'Tariff', 'stdorToU', 'Acorn_grouped', 'file', 'Acorn Category']
+-------+---------+-------------------+--------------+-------------------+------+--------+-------------+-------+-----------------+
|  Acorn|    LCLid|               tstp|energy(kWh/hh)|     TariffDateTime|Tariff|stdorToU|Acorn_grouped|   file|   Acorn Category|
+-------+---------+-------------------+--------------+-------------------+------+--------+-------------+-------+-----------------+
|ACORN-A|MAC000002|2013-01-01 00:00:00|         0.219|2013-01-01 00:00:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 00:30:00|         0.241|2013-01-01 00:30:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 01:00:00|         0.191|2013-01-01 01:00:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 01:30:00|         0.235|2013-01-01 01:30:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 02:00:00|         0.182|2013-01-01 02:00:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 02:30:00|         0.229|2013-01-01 02:30:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 03:00:00|         0.194|2013-01-01 03:00:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 03:30:00|         0.201|2013-01-01 03:30:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 04:00:00|         0.122|2013-01-01 04:00:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
|ACORN-A|MAC000002|2013-01-01 04:30:00|         0.099|2013-01-01 04:30:00|Normal|     Std|     Affluent|block_0|Luxury Lifestyles|
+-------+---------+-------------------+--------------+-------------------+------+--------+-------------+-------+-----------------+
only showing top 10 rows


----------------------------------------------------
After final join with df12 and acorn_energy:
Final columns: ['TariffDateTime', 'Tariff', 'tstp', 'Acorn', 'Acorn_grouped', 'Acorn Category', 'mean_energy']

----------------------------------------------------
Schema of final DataFrame:
root
 |-- TariffDateTime: timestamp (nullable = true)
 |-- Tariff: string (nullable = true)
 |-- tstp: string (nullable = true)
 |-- Acorn: string (nullable = true)
 |-- Acorn_grouped: string (nullable = true)
 |-- Acorn Category: string (nullable = true)
 |-- mean_energy: double (nullable = true)

Number of rows: 139632
+-------------------+------+-------------------+-------+-------------+--------------------+-------------------+
|     TariffDateTime|Tariff|               tstp|  Acorn|Acorn_grouped|      Acorn Category|        mean_energy|
+-------------------+------+-------------------+-------+-------------+--------------------+-------------------+
|2013-01-01 02:00:00|Normal|2013-01-01 02:00:00|ACORN-B|     Affluent|   Luxury Lifestyles|            0.24264|
|2013-01-01 02:00:00|Normal|2013-01-01 02:00:00|ACORN-E|     Affluent|Established Afflu...| 0.1885604895109395|
|2013-01-01 02:00:00|Normal|2013-01-01 02:00:00|ACORN-O|    Adversity|Steadfast Communi...|0.12205825242718446|
|2013-01-01 02:00:00|Normal|2013-01-01 02:00:00|ACORN-G|  Comfortable|Thriving Neighbou...|0.15964215735294118|
|2013-01-01 02:00:00|Normal|2013-01-01 02:00:00|ACORN-I|  Comfortable|Thriving Neighbou...|0.16359183673469388|
+-------------------+------+-------------------+-------+-------------+--------------------+-------------------+
only showing top 5 rows

'''