from pyspark.sql import SparkSession
import boto3
import os

def main():
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("Combine CSV Files") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    # Source and destination paths
    source_bucket = "is459-g1t7-smart-meters-in-london"
    source_prefix = "raw-data/halfhourly_datasets/"
    
    # Temporary directory for the output
    temp_output_dir = "s3://is459-g1t7-smart-meters-in-london/raw-data/temp_halfhourly_dataset/"
    
    # Final destination file
    final_destination = "s3://is459-g1t7-smart-meters-in-london/raw-data/halfhourly_dataset.csv"

    # Use boto3 to list all CSV files in the bucket
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
    
    # Initialize an empty dataframe
    combined_df = None
    
    # Process each file individually
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.endswith('.csv'):
                file_path = f"s3://{source_bucket}/{file_key}"
                print(f"Processing file: {file_path}")
                
                # Read the current CSV file
                current_df = spark.read.option("header", "true").csv(file_path)
                
                # If this is the first file, initialize the combined dataframe
                if combined_df is None:
                    combined_df = current_df
                else:
                    # Union with the combined dataframe
                    combined_df = combined_df.union(current_df)
    
    # If we found and processed files
    if combined_df is not None:
        # Repartition to 1 to get a single output file
        combined_df = combined_df.coalesce(1)
        
        # Write to a temporary directory first
        combined_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(temp_output_dir)
        
        # Now use boto3 to rename the part file to the desired final name
        # First, list files in the temp directory to find the part file
        temp_response = s3.list_objects_v2(
            Bucket=source_bucket, 
            Prefix="raw-data/temp_halfhourly_dataset/"
        )
        
        part_file = None
        for obj in temp_response.get('Contents', []):
            if obj['Key'].endswith('.csv'):
                part_file = obj['Key']
                break
        
        if part_file:
            # Copy the part file to the final destination with the desired name
            s3.copy_object(
                Bucket=source_bucket,
                CopySource={'Bucket': source_bucket, 'Key': part_file},
                Key="raw-data/halfhourly_dataset.csv"
            )
            
            # Delete the temporary directory and its contents
            for obj in temp_response.get('Contents', []):
                s3.delete_object(Bucket=source_bucket, Key=obj['Key'])
            
            print(f"Successfully combined all CSV files and wrote to {final_destination}")
        else:
            print("Error: Could not find the output part file")
    else:
        print("No CSV files found to process")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
