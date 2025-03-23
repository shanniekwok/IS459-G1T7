import requests
import pandas as pd
from datetime import datetime
import boto3
import io

# API configuration
API_KEY = "d41ec021ad5bbf0d628897d8f70f563e" 
# London Coordinates
LAT = 51.5074  
LON = -0.1278  
# API Endpoint (16-Day Forecast)
URL = f"https://pro.openweathermap.org/data/2.5/forecast/daily?lat={LAT}&lon={LON}&cnt=16&units=metric&appid={API_KEY}"

# Fetch Weather Data
response = requests.get(URL)
weather_data = response.json()

# Check if API Request was successful
if response.status_code != 200:
    print(f"Error: {weather_data}")
    exit()

# Process Weather Data
forecast_list = weather_data["list"]

# Extract Relevant Features
weather_records = []
for forecast in forecast_list:
    record = {
        "date": datetime.utcfromtimestamp(forecast["dt"]).strftime('%Y-%m-%d'),  # Extract date only (YYYY-MM-DD)
        "temperatureMax": forecast["temp"]["max"],  
        "temperatureMin": forecast["temp"]["min"],  
        "temperatureHigh": forecast["temp"]["day"],  
        "temperatureLow": forecast["temp"]["night"],  
        "apparentTemperatureHigh": forecast["feels_like"]["day"],  
        "apparentTemperatureLow": forecast["feels_like"]["night"],  
        "apparentTemperatureMin": forecast["feels_like"]["night"],  
        "apparentTemperatureMax": forecast["feels_like"]["day"],  
        "pressure": forecast["pressure"],  
        "humidity": forecast["humidity"],  
        "cloudCover": forecast["clouds"],  
        "windSpeed": forecast["speed"],  
        "windBearing": forecast["deg"],  
        "precipType": forecast.get("weather", [{}])[0].get("main", "Clear"),  
    }
    weather_records.append(record)

# Convert to DataFrame
df_weather = pd.DataFrame(weather_records)

# Encode precipType (Rain, Snow, Clear, etc.)
df_weather["precipType"] = df_weather["precipType"].map({"Rain": 1, "Snow": 2, "Clear": 0}).fillna(0).astype(int)

# Show Data
print(df_weather.head())

# Save to S3
s3_bucket = "is459-g1t7-smart-meters-in-london"
s3_key = "weather-api-results/london_weather_forecast.csv"

# Initialize S3 client
s3_client = boto3.client('s3')

# Convert DataFrame to CSV in memory
csv_buffer = io.StringIO()
df_weather.to_csv(csv_buffer, index=False)

# Upload to S3
s3_client.put_object(
    Bucket=s3_bucket,
    Key=s3_key,
    Body=csv_buffer.getvalue()
)

print(f"Weather data saved to s3://{s3_bucket}/{s3_key}")