import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = "d41ec021ad5bbf0d628897d8f70f563e" 

# **London Coordinates**
LAT = 51.5074  
LON = -0.1278  

# **API Endpoint (16-Day Forecast)**
URL = f"https://pro.openweathermap.org/data/2.5/forecast/daily?lat={LAT}&lon={LON}&cnt=16&units=metric&appid={API_KEY}"

# **Fetch Weather Data**
response = requests.get(URL)
weather_data = response.json()

# **Check if API Request was successful**
if response.status_code != 200:
    print(f"Error: {weather_data}")
    exit()

# **Process Weather Data**
forecast_list = weather_data["list"]

# **Extract Relevant Features**
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

# **Convert to DataFrame**
df_weather = pd.DataFrame(weather_records)

# **Encode precipType (Rain, Snow, Clear, etc.)**
df_weather["precipType"] = df_weather["precipType"].map({"Rain": 1, "Snow": 2, "Clear": 0}).fillna(0).astype(int)

# **Show Data**
print(df_weather.head())

# **Save to weather_API_results folder**
save_path = os.path.join(os.getcwd(), "weather_API_results", "london_weather_forecast.csv")
df_weather.to_csv(save_path, index=False)
print(f"Weather data saved to: {save_path}")