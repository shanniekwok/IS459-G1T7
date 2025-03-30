import os
import pandas as pd
import joblib

# Load the saved weather forecast data
weather_data_path = os.path.join(os.getcwd(), "weather_API_results", "london_weather_forecast.csv")

# Check if file exists
if not os.path.exists(weather_data_path):
    print(f"Error: Weather data file not found at {weather_data_path}")
    exit()

# Load the weather data
df_weather = pd.read_csv(weather_data_path)
print(f"Loaded weather data from {weather_data_path}")
print(df_weather.head())  # Show first 5 rows of the weather data

# Load the trained Random Forest model
model_path = os.path.join(os.getcwd(), "random_forest_model", "randomforest.pkl")  # Adjust path if needed

# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Trained model file not found at {model_path}")
    exit()

# Load model
rf_model = joblib.load(model_path)
print("Random Forest model loaded successfully.")

# Ensure the weather data has the correct feature set
features = [
    "is_weekend", "temp_daylight_interaction", "is_holiday",
    "temperaturemax", "apparenttemperaturemax", "pressure",
    "apparenttemperaturemin", "apparenttemperaturehigh",
    "temperaturehigh", "daylight_duration"
]

# Ensure weather data only includes the required features
df_features = df_weather[features]

# Make predictions
predictions = rf_model.predict(df_features)

# Convert predictions to DataFrame
df_predictions = pd.DataFrame({
    "Date": df_weather["date"],
    "Predicted_Energy_Consumption": predictions
})

# Save predictions to CSV
predictions_save_path = os.path.join(os.getcwd(), "random_forest_output", "london_energy_predictions.csv")
df_predictions.to_csv(predictions_save_path, index=False)
print(f"Predictions saved to {predictions_save_path}")

# Display first 5 predictions
print("\nFirst 5 predictions:")
print(df_predictions.head())