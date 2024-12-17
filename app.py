import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from lab_pred import fetch_openaq_data_parallel, create_sequences, build_multi_output_model
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Streamlit App Title
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
future_steps = st.number_input("Number of Future Steps to Predict", min_value=1, value=24)

# Button to trigger data fetching and prediction
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
                #st.write("Data Preview:", df.tail())

                # Scale the data
                st.write("Preparing data for prediction...")
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df.fillna(0))

                # Prepare input data
                input_data, _ = create_sequences(
                    scaled_data, time_steps=60, target_indices=list(range(len(parameters)))
                )

                # Train LSTM model
                st.write("Training multi-output LSTM model...")
                model = build_multi_output_model(input_shape=input_data.shape[1:], num_outputs=len(parameters))
                model.fit(input_data, input_data[:, -1, :len(parameters)], epochs=5, batch_size=32, verbose=1)

                # Rolling Predictions for Multiple Future Steps
                st.write("Generating multiple future predictions...")
                last_sequence = input_data[-1]  # Get the last input sequence
                future_predictions = []

                for _ in range(future_steps):
                    next_pred = model.predict(last_sequence.reshape(1, *last_sequence.shape))
                    future_predictions.append(next_pred[0])  # Append the predicted step
                    # Update sequence for next prediction
                    last_sequence = np.vstack((last_sequence[1:], next_pred[0]))

                # Inverse-transform predictions to original scale
                inverse_predictions = scaler.inverse_transform(
                    np.hstack((np.array(future_predictions), np.zeros((future_steps, scaled_data.shape[1] - len(parameters)))))
                )[:, :len(parameters)]

                # Generate future timestamps
                future_index = [df.index[-1] + timedelta(hours=i+1) for i in range(future_steps)]
                future_df = pd.DataFrame(inverse_predictions, columns=parameters, index=future_index)
                #st.write("Future Predictions DataFrame:", future_df)

                # Combine actual and predicted data for display
                combined_df = pd.concat([df, future_df])

                # Visualization with Plotly
                st.write("Visualizing Actual and Predicted Data...")
                fig = go.Figure()

                # Plot actual data
                for param in parameters:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[param],
                        mode='lines',
                        name=f"Actual {param.upper()}",
                        line=dict(color='blue')
                    ))

                # Plot predicted data
                for param in parameters:
                    fig.add_trace(go.Scatter(
                        x=future_df.index,
                        y=future_df[param],
                        mode='lines',
                        name=f"Predicted {param.upper()}",
                        line=dict(dash='dash', color='red')
                    ))

                # Update layout for interactivity
                fig.update_layout(
                    title="Actual vs Predicted Air Quality",
                    xaxis_title="Time",
                    yaxis_title="Concentration",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    template="plotly_white"
                )

                # Display the plot
                st.plotly_chart(fig)

            else:
                st.warning("No data available for the selected parameters.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please provide a city and select at least one parameter.")
