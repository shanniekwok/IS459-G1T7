import time
import pandas as pd
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set Up Selenium WebDriver
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Enable headless mode for speed
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Automatically download and use the correct ChromeDriver version
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Base URLs
base_url = "https://www.accuweather.com/en/gb/london/ec4a-2"
urls = [
    f"{base_url}/weather-today/328328",  # Day 1 (Today)
    f"{base_url}/weather-tomorrow/328328"  # Day 2 (Tomorrow)
] + [f"{base_url}/daily-weather-forecast/328328?day={i}" for i in range(3, 30)]  # Days 3-30

# Function to Scrape Weather Data
def scrape_weather():
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'Cloud Cover')]"))
        )

        # Extract Cloud Cover
        cloud_elements = driver.find_elements(By.XPATH, "//p[contains(text(), 'Cloud Cover')]/span[@class='value']")
        cloud_cover_values = [elem.text.strip().replace("%", "") for elem in cloud_elements]

        if len(cloud_cover_values) == 2:
            day_cloud = int(cloud_cover_values[0])
            night_cloud = int(cloud_cover_values[1])
            avg_cloud_cover = (day_cloud + night_cloud) / 2
        elif len(cloud_cover_values) == 1:
            avg_cloud_cover = int(cloud_cover_values[0])
        else:
            avg_cloud_cover = None

        # Extract Max & Min Temperature
        try:
            max_temp_element = driver.find_element(By.XPATH, "//div[contains(@class, 'temp-history')]//div[@class='temperature'][1]").text.strip()
            min_temp_element = driver.find_element(By.XPATH, "//div[contains(@class, 'temp-history')]//div[@class='temperature'][2]").text.strip()
            max_temp = int(max_temp_element.replace("°", ""))
            min_temp = int(min_temp_element.replace("°", ""))
        except:
            max_temp, min_temp = None, None

        # Extract Date
        try:
            date_element = driver.find_element(By.XPATH, "//div[contains(@class, 'subnav-pagination')]/div").text.strip()
        except:
            date_element = "Unknown Date"

        return {"Date": date_element, "CloudCover": avg_cloud_cover, "MaxTemp": max_temp, "MinTemp": min_temp}

    except Exception as e:
        print(f"Error scraping weather data: {e}")
        return None

# Scrape Data for Each Day
weather_data = []
for url in urls:
    driver.get(url)
    print(f"Scraping: {url}")
    time.sleep(2)  # Reduce delay for faster execution
    data = scrape_weather()
    if data:
        weather_data.append(data)
        print(f"Date: {data['Date']}, Cloud Cover: {data['CloudCover']}%, Max Temp: {data['MaxTemp']}°C, Min Temp: {data['MinTemp']}°C")

# Close the WebDriver
driver.quit()

# Convert Data to Pandas DataFrame
weather_df = pd.DataFrame(weather_data)
print(weather_df.head(10))  # Use head() instead of show()

# Save CSV to "scraping_results" folder
repo_root = os.getcwd()
output_folder = os.path.join(repo_root, "scraping_results")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "london_cloud_cover_forecast.csv")
weather_df.to_csv(output_file, index=False)

print(f"Saved weather data to {output_file}")