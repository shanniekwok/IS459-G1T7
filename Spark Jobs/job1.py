from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# S3 Paths
S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/"

def main():
    # Create Spark Session
    spark = SparkSession.builder.appName("SmartMetersProcessing").getOrCreate()

    # Read input CSV files from S3
    df1 = spark.read.csv(f"{S3_INPUT_FOLDER}daily_dataset.csv", header=True, inferSchema=True)
    df5 = spark.read.csv(f"{S3_INPUT_FOLDER}informations_households.csv", header=True, inferSchema=True)
    df10_1 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_information.csv", header=True, inferSchema=True)
    df10_2 = spark.read.csv(f"{S3_INPUT_FOLDER}acorn_category_information.csv", header=True, inferSchema=True)

    # Merge df10_1 and df10_2 on "Acorn Category"
    df10 = df10_1.join(df10_2, on="Acorn Category", how="left")

    # Merge df1 and df5 on "LCLid"
    merged_df1_df5 = df1.join(df5, on="LCLid", how="left")

    # Merge the above result with df10 on "Acorn"
    final_merged_df = merged_df1_df5.join(df10, on="Acorn", how="left")

    # Print the number of records after merging
    print("Total number of records after merging:", final_merged_df.count())

    # Write the final merged dataset to S3 in **Parquet format**
    final_merged_df.write.mode("overwrite").parquet(f"{S3_OUTPUT_FOLDER}merged_df1_df5_df10")

    print("Merged dataset successfully written to S3 as Parquet:", S3_OUTPUT_FOLDER)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()