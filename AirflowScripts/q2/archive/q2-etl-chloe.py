# Datasets:
# df1 daily_dataset.zip (not csv) - block files of a household's daily energy consumption
# df7 uk_bank_holidays.csv 
# df8 weather_daily_darksky.csv
# Output: q2_daily_energy_weather (Parquet format) -> df1 + df8 + df7

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, to_timestamp, unix_timestamp, date_format, col, substring, when
from pyspark.sql.types import DateType
# from pyspark.sql.window import Window

def create_spark_session():
    """
    Create a Spark session with appropriate configurations
    """
    return SparkSession.builder \
        .appName("IS459-Q2-ETL") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

def read_csv_from_s3(spark, s3_path, file_name):
    """
    Read CSV file from S3 bucket
    
    :param spark: SparkSession
    :param s3_path: S3 input folder path
    :param file_name: Name of the CSV file
    :return: DataFrame
    """
    return spark.read.csv(f"{s3_path}{file_name}", header=True, inferSchema=True)

def parse_and_join_dataframes(df1, df7, df8):
    """
    Parse dates and join dataframes
    
    :param df1: Daily Energy DataFrame (daily_dataset.csv)
    :param df7: Daily Weather DataFrame (weather_daily_darksky.csv)
    :param df8: Bank Holidays DataFrame (uk_bank_holidays.csv)
    :param s3_output_folder: S3 output folder path
    """
    # Parse date from daily_dataset.csv (format: dd/mm/yyyy)
    df1_with_date = df1.withColumn("parsed_date", 
        to_date(col("day"), "dd/MM/yyyy")
    )

    # print first few rows of df1_with_date for debugging
    df1_with_date.show(5)
    
    # Parse date from weather_daily_darksky.csv (format: dd/mm/yyyy HH:mm)

    # Split "time" column to get the date part a space separates the date and time
    # make the date into "date" and time into "time"
    df7_with_date = df7.withColumn("date", substring(col("time"), 1, 10)) \
    .withColumn("time", substring(col("time"), 12, 5))

    # Convert sunrise and sunset time strings to timestamps
    df7 = df7.withColumn("sunriseTime", to_timestamp(col("sunriseTime"), "MM/dd/yyyy h:mm:ss a")) \
    .withColumn("sunsetTime", to_timestamp(col("sunsetTime"), "MM/dd/yyyy h:mm:ss a"))

    # print first few rows of df7_with_date for debugging
    df7_with_date.show(5)

    # Parse dates for bank holidays
    df8_with_date = df8

    # print first few rows of df8_with_date for debugging
    df8_with_date.show(5)
    
    # Join dataframes
    merged_df = df1_with_date.join(
        df7_with_date, 
        df1_with_date.parsed_date == df7_with_date.date, 
        "left"
    )

    # Add holiday information
    merged_df = merged_df.join(
        df8_with_date.select(col("Bank holidays"), col("Type")),
        df1_with_date.parsed_date == df8_with_date["Bank holidays"],
        "left"
    ).withColumn("holiday", 
        when(col("Type").isNotNull(), col("Type"))
        .otherwise("No Holiday")
    )

    # Drop duplicate date column and select desired columns
    merged_df = merged_df.drop(df1_with_date.parsed_date ,df7_with_date.date, df7_with_date.time, df8["Bank holidays"], df8_with_date.Type) \
    
    # impute missing values "temperatureMax", "temperatureMin", "humidity", "cloudCover" with their mean
    merged_df = merged_df.fillna({
        "temperatureMax": merged_df.select("temperatureMax").agg({"temperatureMax": "avg"}).first()[0],
        "temperatureMin": merged_df.select("temperatureMin").agg({"temperatureMin": "avg"}).first()[0],
        "humidity": merged_df.select("humidity").agg({"humidity": "avg"}).first()[0],
        "cloudCover": merged_df.select("cloudCover").agg({"cloudCover": "avg"}).first()[0]
    })

    # Derive time-based features for visualisation from "day" (in dd/mm/yyyy format)
    # 1. "day_of_week" = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    # 2. "month" = "January", "February", ..., "December"
    # 3. "season" = "Winter", "Spring", "Summer", "Autumn"
    # 4. "is_weekend" = 1 (True) OR 0 (False)
    # 5. "is_holiday" = 1 (True) OR 0 (False) (if "holiday" is not "No Holiday")

    # Derive weather-based features to help with ML prediction of energy consumption
    # 1. "temp_variation" = "temperatureMax" - "temperatureMin"
    # 2. "temp_humidity_interaction" = "humidity" * "temperatureMax"
    # 3. "temp_cloudcover_interaction" = "cloudCover" * "temperatureMax"
    # 4. "temp_uvindex_interaction" = "uvIndex" * "temperatureMax" (measuring sun exposure impact)
    # 5. "weekend_energy_interaction" = "is_weekend" * "energy_mean"
    # 6. "holiday_energy_interaction" = "is_holiday" * "energy_mean"
    # 7. "daylight_duration" = "sunsetTime" - "sunriseTime"
    # 8. "temp_daylight_interaction" = "daylight_duration" * "temperatureMax" (hot & long days could lead to more AC usage)

    merged_df = merged_df.withColumn("day_of_week", date_format(to_date(col("day"), "dd/MM/yyyy"), "EEEE")) \
    .withColumn("month", date_format(to_date(col("day"), "dd/MM/yyyy"), "MMMM")) \
    .withColumn("season", when(col("month").isin("December", "January", "February"), "Winter")
    .when(col("month").isin("March", "April", "May"), "Spring")
    .when(col("month").isin("June", "July", "August"), "Summer")
    .otherwise("Autumn")) \
    .withColumn("is_weekend", when(col("day_of_week").isin("Saturday", "Sunday"), 1).otherwise(0)) \
    .withColumn("is_holiday", when(col("holiday") != "No Holiday", 1).otherwise(0)) \
    .withColumn("temp_variation", col("temperatureMax") - col("temperatureMin")) \
    .withColumn("temp_humidity_interaction", col("humidity") * col("temperatureMax")) \
    .withColumn("temp_cloudcover_interaction", col("cloudCover") * col("temperatureMax")) \
    .withColumn("temp_uvindex_interaction", col("uvIndex") * col("temperatureMax")) \
    .withColumn("weekend_energy_interaction", col("is_weekend") * col("energy_mean")) \
    .withColumn("holiday_energy_interaction", col("is_holiday") * col("energy_mean")) \
    .withColumn("daylight_duration", (unix_timestamp(col("sunsetTime")) - unix_timestamp(col("sunriseTime")))/3600) \
    .withColumn("temp_daylight_interaction", col("daylight_duration") * col("temperatureMax"))

    # Create a window specification for the previous day
    # window_spec = Window.orderBy("day")
    # # Add the previous day's energy sum
    # merged_df = merged_df.withColumn("prev_day_energy_sum", lag("energy_sum", 1).over(window_spec))
    
    return merged_df

def main():
    # S3 paths
    S3_INPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/raw-data/"
    S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/q2_daily_energy_weather_inner"
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Read input CSVs
        df1 = read_csv_from_s3(spark, S3_INPUT_FOLDER, "daily_dataset.csv")
        df7 = read_csv_from_s3(spark, S3_INPUT_FOLDER, "weather_daily_darksky.csv")
        df8 = read_csv_from_s3(spark, S3_INPUT_FOLDER, "uk_bank_holidays.csv")

        # Parse and join dataframes
        merged_df = parse_and_join_dataframes(df1, df7, df8)
        
        # Show sample of merged dataframe
        merged_df.show()

        print("Number of rows in merged dataframe: ", merged_df.count())

        # Write to Parquet
        merged_df.write.mode("overwrite").parquet(S3_OUTPUT_FOLDER)
        
        print(f"Merged dataframe successfully written to {S3_OUTPUT_FOLDER}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()