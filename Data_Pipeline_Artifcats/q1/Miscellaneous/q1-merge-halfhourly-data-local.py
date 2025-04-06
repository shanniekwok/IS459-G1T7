from pyspark.sql import SparkSession
import os
import shutil

def main():
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("Combine CSV Files") \
        .master("local[*]") \
        .config("spark.security.manager", "false") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()

    # Source and destination paths
    source_file_dir = "../../Data/halfhourly_dataset"  # Local directory containing CSV files
    temp_output_dir = "../../Temp Data/"  # Temporary directory for output
    final_destination = "../../Data/halfhourly_dataset.csv"  # Final destination file
    
    # Initialize an empty dataframe
    combined_df = None
    
    # Loop through all CSV files in the source directory
    for root, dirs, files in os.walk(source_file_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
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
        
        # Write to the temporary directory first
        combined_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(temp_output_dir)
        
        # Now, move the combined file to the final destination (get the part file)
        part_file = None
        for root, dirs, files in os.walk(temp_output_dir):
            for file in files:
                if file.endswith('.csv'):
                    part_file = os.path.join(root, file)
                    break
        
        if part_file:
            # Rename the part file to the final destination
            os.rename(part_file, final_destination)
            
            # Clean up: Delete the temporary directory and its contents
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)

            print(f"Successfully combined all CSV files and wrote to {final_destination}")
        else:
            print("Error: Could not find the output part file")
    else:
        print("No CSV files found to process")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()