import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from lab_pred import fetch_openaq_data_parallel, create_sequences, build_multi_output_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Streamlit App
st.title("Air Quality Monitoring and Prediction")

# User Inputs
city = st.text_input("Enter City Name", placeholder="e.g., London")
parameters = st.multiselect(
    "Select Parameters to Predict",
    options=["pm25", "pm10", "no2", "o3", "co", "so2"],
    default=["pm25", "pm10"]
)
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.date_input("End Date", value=datetime.now())
predict_future = st.checkbox("Predict Future Air Quality", value=True)

if st.button("Fetch and Predict Data"):
    if not API_KEY:
        st.error("API key is missing. Please configure it in the .env file.")
    elif city and parameters:
        st.write(f"Fetching data for {city} from {start_date} to {end_date}...")
        try:
            # Fetch historical data
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

                if predict_future:
                    # Prepare data for prediction
                    st.write("Preparing data for prediction...")
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df.fillna(0))
                    input_data, target_data = create_sequences(
                        scaled_data, time_steps=60, target_indices=list(range(len(parameters)))
                    )

                    # Load or build LSTM model
                    model = build_multi_output_model(input_shape=input_data.shape[1:], num_outputs=len(parameters))
                    st.write("Training multi-output LSTM model...")
                    model.fit(input_data, target_data, epochs=5, batch_size=32, verbose=1)

                    # Predict future air quality
                    st.write("Predicting future air quality...")
                    predictions = model.predict(input_data[-1].reshape(1, *input_data[-1].shape))

                    # Visualize predictions
                    st.write("Visualizing predictions...")
                    plt.figure(figsize=(12, 6))
                    for i, param in enumerate(parameters):
                        plt.plot(df.index[-60:], df[param].iloc[-60:], label=f"Actual {param.upper()}")
                        plt.plot([df.index[-1] + timedelta(hours=i) for i in range(1, len(predictions[0]) + 1)],
                                 predictions[0][:, i], label=f"Predicted {param.upper()}", linestyle="--")
                    plt.title("Actual vs. Predicted Air Quality")
                    plt.xlabel("Time")
                    plt.ylabel("Concentration")
                    plt.legend()
                    plt.grid()
                    st.pyplot(plt)
            else:
                st.warning("No data available for the selected parameters.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please provide a city and select at least one parameter.")
