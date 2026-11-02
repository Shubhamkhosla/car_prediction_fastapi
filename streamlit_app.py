
import streamlit as st
import requests
import pandas as pd
import os

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict")

st.title("🚗 Used Car Price Prediction")
st.write("Predict the selling price of used cars using XGBoost.")

st.divider()

# --------------------------------
# Number of cars
# --------------------------------

number_of_cars = st.number_input(
    "How many cars do you want to predict?",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)

cars = []

# --------------------------------
# Car input form
# --------------------------------

with st.form("car_prediction_form"):

    for i in range(int(number_of_cars)):

        st.subheader(f"Car {i + 1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            brand = st.text_input(
                "Brand",
                value="Maruti",
                key=f"brand_{i}"
            )

            model = st.text_input(
                "Model",
                value="Alto",
                key=f"model_{i}"
            )

            vehicle_age = st.number_input(
                "Vehicle Age (years)",
                min_value=0,
                max_value=50,
                value=5,
                key=f"age_{i}"
            )

            km_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                value=45000,
                key=f"km_{i}"
            )

        with col2:
            seller_type = st.selectbox(
                "Seller Type",
                ["Individual", "Dealer", "Trustmark Dealer"],
                key=f"seller_{i}"
            )

            fuel_type = st.selectbox(
                "Fuel Type",
                ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                key=f"fuel_{i}"
            )

            transmission_type = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"],
                key=f"transmission_{i}"
            )

            mileage = st.number_input(
                "Mileage",
                min_value=0.1,
                value=20.5,
                key=f"mileage_{i}"
            )

        with col3:
            engine = st.number_input(
                "Engine (CC)",
                min_value=1,
                value=998,
                key=f"engine_{i}"
            )

            max_power = st.number_input(
                "Max Power",
                min_value=0.1,
                value=67.1,
                key=f"power_{i}"
            )

            seats = st.number_input(
                "Seats",
                min_value=1,
                max_value=20,
                value=5,
                key=f"seats_{i}"
            )

        cars.append({
            "brand": brand,
            "model": model,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        })

        st.divider()

    submitted = st.form_submit_button(
        "Predict Car Prices 🚀",
        use_container_width=True
    )


# --------------------------------
# Send request to FastAPI
# --------------------------------

if submitted:

    try:
        with st.spinner("Predicting car prices..."):

            response = requests.post(
                API_URL,
                json=cars,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Predictions generated successfully!")

            st.subheader("Prediction Results")

            predictions = result["predictions"]

            result_df = pd.DataFrame(predictions)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            # Display individual predicted prices
            for i, prediction in enumerate(predictions):

                st.metric(
                    label=f"{prediction['brand']} {prediction['model']}",
                    value=f"₹{prediction['predicted_price']:,.2f}"
                )

        else:
            st.error(
                f"API Error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start your backend first."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
