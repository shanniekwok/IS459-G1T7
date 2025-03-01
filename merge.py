from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("MergeCSVFiles").getOrCreate()

# Load CSV files into DataFrames
df1 = spark.read.csv("s3://your-bucket/daily_dataset.csv", header=True, inferSchema=True)
df5 = spark.read.csv("s3://your-bucket/informations_households.csv", header=True, inferSchema=True)
df10 = spark.read.csv("s3://your-bucket/acorn_info.csv", header=True, inferSchema=True)
df11 = spark.read.csv("s3://your-bucket/acorn_category_info.csv", header=True, inferSchema=True)

# Merge df1 and df5 on 'LCLid'
merged_df1 = df1.join(df5, on="LCLid", how="inner")

# Merge df10 and df11 on 'Acorn Category'
merged_df2 = df10.join(df11, on="Acorn Category", how="inner")

# Merge the two merged datasets on 'Acorn'
final_df = merged_df1.join(merged_df2, on="Acorn", how="inner")

# Save the final merged DataFrame to S3
final_df.write.csv("s3://your-bucket/merged_output.csv", header=True)

# Stop Spark session
spark.stop()
