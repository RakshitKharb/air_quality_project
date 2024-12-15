import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lab_pred import fetch_openaq_data_parallel
from dotenv import load_dotenv
import os

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Streamlit App
st.title("Air Quality Monitoring and Prediction")
st.write("Monitor and visualize air quality data using OpenAQ API.")

# User Inputs
city = st.text_input("Enter City Name", placeholder="e.g., London")
parameters = st.multiselect(
    "Select Parameters to Monitor",
    options=["pm25", "pm10", "no2", "o3", "co", "so2"],
    default=["pm25", "pm10"]
)
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
end_date = st.date_input("End Date", value=datetime.now())

if st.button("Fetch Air Quality Data"):
    if not API_KEY:
        st.error("API key is missing. Please configure it in the .env file.")
    elif city and parameters:
        st.write(f"Fetching data for {city} from {start_date} to {end_date}...")
        try:
            df = fetch_openaq_data_parallel(
                city=city,
                parameters=parameters,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                api_key=API_KEY
            )
            if df is not None and not df.empty:
                st.success("Data fetched successfully!")
                st.write("Data Preview:", df.head())
                
                # Visualization
                st.write("Air Quality Trends:")
                plt.figure(figsize=(12, 6))
                for param in parameters:
                    if param in df.columns:
                        plt.plot(df.index, df[param], label=param.upper())
                plt.title(f"Air Quality Trends in {city}")
                plt.xlabel("Time")
                plt.ylabel("Concentration")
                plt.legend()
                plt.grid()
                st.pyplot(plt)
            else:
                st.warning("No data available for the selected parameters.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
    else:
        st.warning("Please provide a city name and select at least one parameter.")
