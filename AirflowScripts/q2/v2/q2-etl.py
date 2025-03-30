import pandas as pd
import boto3
from io import BytesIO, StringIO

# S3 bucket paths
S3_DAILY_DATASET = "s3://is459-g1t7-smart-meters-in-london/raw-data/daily_dataset.csv"
S3_WEATHER_DATA = "s3://is459-g1t7-smart-meters-in-london/raw-data/weather_daily_darksky.csv"
S3_UK_BANK_HOLIDAYS = "s3://is459-g1t7-smart-meters-in-london/raw-data/uk_bank_holidays.csv"
S3_OUTPUT_FOLDER = "s3://is459-g1t7-smart-meters-in-london/processed-data/merged_daily_weather_data/"

# Initialize S3 client
s3 = boto3.client('s3')

def read_csv_from_s3(s3_path):
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Load the data from S3
print("Reading datasets from S3...")
daily_dataset = read_csv_from_s3(S3_DAILY_DATASET)
weather_data = read_csv_from_s3(S3_WEATHER_DATA)
uk_bank_holidays = read_csv_from_s3(S3_UK_BANK_HOLIDAYS)

weather_data['date'] = pd.to_datetime(weather_data['time'], format='%d/%m/%Y %H:%M', dayfirst=True).dt.strftime('%d/%m/%Y')

# Impute missing values in 'cloudcover' with the mean of the column
weather_data['cloudCover'].fillna(weather_data['cloudCover'].mean(), inplace=True)
# impute missing values in 'uvIndex' with 0
weather_data['uvIndex'].fillna(0, inplace=True)

# drop uvIndexTime
weather_data.drop(columns=['uvIndexTime'], inplace=True)

# Ensure the 'day' column in daily_dataset is in the correct format
# It should already be in DD/MM/YYYY format based on the sample
# But we'll handle potential mixed formats just in case
# try:
    # First try with the expected format
# daily_dataset['day'] = pd.to_datetime(daily_dataset['day'], 
                                        #  format='%d/%m/%Y').dt.strftime('%d/%m/%Y')
# except ValueError:
#     # If that fails, try with a more flexible approach
daily_dataset['day'] = pd.to_datetime(daily_dataset['day'], 
                                        format='mixed', 
                                        dayfirst=True).dt.strftime('%d/%m/%Y')

# Force join where the date is "27/10/2013" and "28/10/2012"

# Now we can merge the dataframes on day and date
merged_df = pd.merge(daily_dataset, weather_data, left_on='day', right_on='date', how='left')
print("number of rows in merged_df:", len(merged_df))

# Drop rows with any missing values in the merged dataframe
merged_df.dropna(inplace=True)
print("number of rows in merged_df after dropping missing values:", len(merged_df))

# Format the 'Bank holidays' column in the uk_bank_holidays dataframe to 'DD/MM/YYYY'
uk_bank_holidays['Bank holidays'] = pd.to_datetime(uk_bank_holidays['Bank holidays'], format='mixed', dayfirst=True).dt.strftime('%d/%m/%Y')

# Match the 'Bank holidays' column with the 'date' column in the merged dataframe --> put the 'Type' column in the merged dataframe and match the value, but put 'No Holiday' if there is no match
merged_df = pd.merge(merged_df, uk_bank_holidays, 
                      left_on='day', right_on='Bank holidays', how='left')

# Fill NaN values in the 'Type' column with 'No Holiday'
merged_df['Type'].fillna('No Holiday', inplace=True)

# Drop the 'Bank holidays' column from the merged dataframe
merged_df.drop(columns=['Bank holidays'], inplace=True)

print("number of rows in merged_df after merging with bank holidays:", len(merged_df))
# print null values for all columns in the merged dataframe
print(merged_df.isnull().sum())

  # Derive time-based features for visualisation from "day" (in dd/mm/yyyy format)
# 1. "day_of_week" = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
merged_df['day'] = pd.to_datetime(merged_df['day'], format='%d/%m/%Y')
merged_df['day_of_week'] = merged_df['day'].dt.day_name()

# 2. "month" = "January", "February", ..., "December"
merged_df['month'] = merged_df['day'].dt.month_name()

# 3. "season" = "Winter", "Spring", "Summer", "Autumn" --> London seasons
#    Winter: December, January, February
#    Spring: March, April, May
#    Summer: June, July, August
#    Autumn: September, October, November
# Map 'month' to 'season'
season_map = {
    'January': 'Winter',
    'February': 'Winter',
    'March': 'Spring',
    'April': 'Spring',
    'May': 'Spring',
    'June': 'Summer',
    'July': 'Summer',
    'August': 'Summer',
    'September': 'Autumn',
    'October': 'Autumn',
    'November': 'Autumn',
    'December': 'Winter'
}
merged_df['season'] = merged_df['month'].map(season_map)

# 4. "is_weekend" = 1 (True) OR 0 (False)
#    Saturday and Sunday are considered weekends
merged_df['is_weekend'] = merged_df['day'].dt.dayofweek >= 5

# 5. "is_holiday" = 1 (True) OR 0 (False) (if "holiday" is not "No Holiday")
#    All other holidays are considered holidays
merged_df['is_holiday'] = merged_df['Type'].apply(lambda x: 1 if x != 'No Holiday' else 0)

# Derive weather-based features to help with ML prediction of energy consumption
# 1. "temp_variation" = "temperatureMax" - "temperatureMin"
merged_df['temp_variation'] = merged_df['temperatureMax'] - merged_df['temperatureMin']

# 2. "temp_humidity_interaction" = "humidity" * "temperatureMax"
merged_df['temp_humidity_interaction'] = merged_df['humidity'] * merged_df['temperatureMax']

# 3. "temp_cloudcover_interaction" = "cloudCover" * "temperatureMax"
merged_df['temp_cloudcover_interaction'] = merged_df['cloudCover'] * merged_df['temperatureMax']

# 4. "temp_uvindex_interaction" = "uvIndex" * "temperatureMax" (measuring sun exposure impact)
merged_df['temp_uvindex_interaction'] = merged_df['uvIndex'] * merged_df['temperatureMax']

# 5. "weekend_energy_interaction" = "is_weekend" * "energy_mean"
merged_df['weekend_energy_interaction'] = merged_df['is_weekend'] * merged_df['energy_mean']

# 6. "holiday_energy_interaction" = "is_holiday" * "energy_mean"
merged_df['holiday_energy_interaction'] = merged_df['is_holiday'] * merged_df['energy_mean']


# 7. "daylight_duration" = "sunsetTime" - "sunriseTime"
# Get the time from the 'sunriseTime' and 'sunsetTime' columns (current format is "dd/mm/yyyy hh:mm")
# Convert to datetime format
merged_df['sunriseTime'] = pd.to_datetime(merged_df['sunriseTime'], format='%d/%m/%Y %H:%M')
merged_df['sunsetTime'] = pd.to_datetime(merged_df['sunsetTime'], format='%d/%m/%Y %H:%M')
# Calculate the duration of daylight in hours
merged_df['daylight_duration'] = (merged_df['sunsetTime'] - merged_df['sunriseTime']).dt.total_seconds() / 3600


# 8. "temp_daylight_interaction" = "daylight_duration" * "temperatureMax" (hot & long days could lead to more AC usage)
merged_df['temp_daylight_interaction'] = merged_df['daylight_duration'] * merged_df['temperatureMax']


print("number of rows in merged_df after feature:", len(merged_df))
# print null values for all columns in the merged dataframe
print(merged_df.isnull().sum())

# Save to Parquet and upload to S3 --> merged_df.write.mode("overwrite").parquet(S3_OUTPUT_FOLDER)
# Convert the DataFrame to Parquet format
# Convert the DataFrame to Parquet format
parquet_buffer = BytesIO()
merged_df.to_parquet(parquet_buffer, index=False)

# Upload the Parquet file to S3
s3.put_object(Bucket="is459-g1t7-smart-meters-in-london", 
              Key="processed-data/merged_daily_weather_data/merged.parquet", 
              Body=parquet_buffer.getvalue())

print("Merge completed and saved to S3 as Parquet.")