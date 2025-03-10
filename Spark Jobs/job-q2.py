# Datasets needed: 
# df1 daily_dataset.csv
# df2 halfhourly_dataset.csv (will need to change df1 -> df2)
# df3 hhblock_dataset
# df7 uk_bank_holidays.csv 
# df8 weather_daily_darksky.csv
# df9 weather_hourly_darksky.csv (will need to change df8 -> df9)
# Output: merged_df1_df3_df7_df8 (Parquet format)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format

# S3 Paths
# S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
# S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessing").getOrCreate()

    # Read input CSV files from S3
    df1 = spark.read.csv(f"{S3_INPUT_FOLDER}daily_dataset.csv", header=True, inferSchema=True)
    df7 = spark.read.csv(f"{S3_INPUT_FOLDER}uk_bank_holidays.csv", header=True, inferSchema=True)
    df8 = spark.read.csv(f"{S3_INPUT_FOLDER}weather_daily_darksky.csv", header=True, inferSchema=True)
    df9 = spark.read.csv(f"{S3_INPUT_FOLDER}weather_hourly_darksky.csv", header=True, inferSchema=True)

    # Extract date from "time" column in df8 and format it to match df1's "day" column format
    df8 = df8.withColumn("day", 
                        date_format(
                            to_date(col("time"), "MM/dd/yyyy"), 
                            "yyyy-MM-dd"
                        ))
    
    # Merge df1 and df8 on "day" column
    merged_df1_df8 = df1.join(df8, on="day", how="left")

    # Process df7 to convert "Bank holidays" column to the format matching df1's "day" column
    df7 = df7.withColumnRenamed("Bank holidays", "holiday_date")
    df7 = df7.withColumn("day", 
                        date_format(
                            to_date(col("holiday_date"), "dd/MM/yyyy"), 
                            "yyyy-MM-dd"
                        ))
    
    # Merge df7 with the previously merged dataframe and only add the "Type" column if the days overlap
    final_merged_df = merged_df1_df8.join(
        df7.select("day", "Type"), 
        on="day", 
        how="left"
    )

    # Print the number of records after merging
    print("Total number of records after merging:", final_merged_df.count())

    # Write the final merged dataset to S3 in **Parquet format**
    final_merged_df.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df1_df3_df7_df8")

    print("Merged dataset successfully written to S3 as Parquet:", S3_OUTPUT_FOLDER)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()