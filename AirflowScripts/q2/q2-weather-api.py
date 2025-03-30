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
    dt = datetime.utcfromtimestamp(forecast["dt"])
    is_weekend = 1 if dt.weekday() >= 5 else 0
    is_holiday = 0  # no holiday info in OpenWeather API

    # Sunrise and sunset for daylight calculation
    sunrise = forecast.get("sunrise")
    sunset = forecast.get("sunset")
    daylight_duration = (sunset - sunrise) / 3600 if sunset and sunrise else None  # in hours

    # Feels like calculations
    feels_like = forecast["feels_like"]
    apparent_max = max(feels_like.values())
    apparent_min = min(feels_like.values())
    apparent_high = feels_like.get("day")  # during daytime

    temperaturemax = forecast["temp"]["max"]
    temperaturehigh = forecast["temp"]["day"]

    record = {
        "date": dt.strftime('%Y-%m-%d'),
        "is_weekend": is_weekend,
        "temp_daylight_interaction": temperaturemax * daylight_duration if daylight_duration else None,
        "is_holiday": is_holiday,
        "temperaturemax": temperaturemax,
        "apparenttemperaturemax": apparent_max,
        "apparenttemperaturemin": apparent_min,
        "apparenttemperaturehigh": apparent_high,
        "temperaturehigh": temperaturehigh,
        "pressure": forecast["pressure"],
        "daylight_duration": daylight_duration,
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