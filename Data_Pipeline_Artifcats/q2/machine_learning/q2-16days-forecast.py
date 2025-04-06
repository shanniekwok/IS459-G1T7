import os
import pandas as pd
import joblib
import dateutil  # Add this explicit import
import boto3
import io
import tempfile

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'is459-g1t7-smart-meters-in-london'

# Load the saved weather forecast data from S3
weather_s3_path = 'weather-api-results/london_weather_forecast.csv'
print(f"Loading weather data from s3://{bucket_name}/{weather_s3_path}")

try:
    # Get the object from S3
    obj = s3.get_object(Bucket=bucket_name, Key=weather_s3_path)
    # Read the CSV directly from the S3 object
    df_weather = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print("Weather data loaded successfully")
    print(df_weather.head())  # Show first 5 rows of the weather data
except Exception as e:
    print(f"Error loading weather data: {e}")
    exit()

# Load the trained Random Forest model from S3
model_s3_path = 'ml-models/randomforest.pkl'
print(f"Loading model from s3://{bucket_name}/{model_s3_path}")

try:
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile() as tmp:
        # Download the model file from S3 to the temporary file
        s3.download_fileobj(Bucket=bucket_name, Key=model_s3_path, Fileobj=tmp)
        # Seek to the beginning of the file
        tmp.seek(0)
        # Load the model
        rf_model = joblib.load(tmp.name)
    print("Random Forest model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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

# Save predictions to S3
output_s3_path = 'ml-output/london_energy_predictions.csv'
print(f"Saving predictions to s3://{bucket_name}/{output_s3_path}")

try:
    # Convert DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    df_predictions.to_csv(csv_buffer, index=False)
    
    # Upload the CSV to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=output_s3_path,
        Body=csv_buffer.getvalue()
    )
    print(f"Predictions saved to s3://{bucket_name}/{output_s3_path}")
except Exception as e:
    print(f"Error saving predictions: {e}")

# Display first 5 predictions
print("\nFirst 5 predictions:")
print(df_predictions.head())