import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load CSV file into pandas DataFrame
df = pd.read_csv("../Data/merged_df.csv")

# Define features and target variable
features1 = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "pressure", "humidity", "cloudCover", "windSpeed", "windBearing", "precipType"
]

features2 = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "humidity", "cloudCover"
]

features3 = [
    "temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow",
    "humidity", "cloudCover", "precipType"
]

features4 = [
    "apparentTemperatureHigh", "apparentTemperatureLow", "apparentTemperatureMin", "apparentTemperatureMax",
    "humidity", "cloudCover", "precipType"
]

features5 = [
    "pressure", "humidity", "cloudCover", "windSpeed", "windBearing", "precipType"
]

# Define initial features (all numeric + encoded categorical)
# numeric_columns = df.select_dtypes(include=['number']).columns
# categorical_columns = df.select_dtypes(include=['object', 'category']).columns
# features = list(numeric_columns) + list(categorical_columns)

target = "energy_mean"

# Identify numeric and categorical columns
numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Impute numeric columns with the mean
num_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

# Encode categorical variables using LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Check for remaining missing values (optional)
print("Missing values after imputation:")
print(df.isnull().sum())

# Split into train and test sets using sklearn's train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train the Random Forest model
X_train = train_df[features5]
y_train = train_df[target]
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model on test data
X_test = test_df[features5]
y_test = test_df[target]

predictions = rf_model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
print(f"RF RMSE: {rmse}")

# Display a sample of predictions
sample_size = min(20, len(predictions))
predictions_df = pd.DataFrame({"Actual": y_test[:sample_size].values, "Predicted": predictions[:sample_size]})
print(predictions_df)

# Extract feature importances
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features5,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(importance_df)
