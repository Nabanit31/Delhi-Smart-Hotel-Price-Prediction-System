import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------
model = pickle.load(open("regression_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Delhi Hotel Price Predictor",
    layout="wide"
)

st.title("🏨 Delhi Smart Hotel Price Estimator for a Week")
st.markdown(
    "Predicts hotel prices using a Gradient Boosting model trained on Delhi hotel data."
)

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Hotel Details")

# --- Categorical Inputs (Label Encoded) ---
district = st.sidebar.selectbox(
    "District",
    label_encoders["District"].classes_
)

transport = st.sidebar.selectbox(
    "Transportation Facility",
    label_encoders["transportation_Facitities"].classes_
)

# --- Room Category (Grouped for UI) ---
room_type = st.sidebar.selectbox(
    "Room Category",
    [
        "Standard/Double",
        "Deluxe/Superior",
        "Premium/Executive",
        "Family/Triple",
        "Suite/Luxury",
        "Dormitory"
    ]
)

# --- Numerical Inputs ---
score = st.sidebar.slider("Customer Score", 0.0, 10.0, 8.0)
reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=50)
beds = st.sidebar.number_input("Number of Beds", min_value=1, value=1)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
def predict_price():
    # Create empty dataframe with correct feature order
    X = pd.DataFrame(0, index=[0], columns=features)

    # --- Label Encoding ---
    X["District"] = label_encoders["District"].transform([district])[0]
    X["transportation_Facitities"] = (
        label_encoders["transportation_Facitities"].transform([transport])[0]
    )

    # --- Numerical Features ---
    X["score"] = score
    X["reviews"] = reviews
    X["total_beds_log"] = np.log1p(beds)

    # --- Features used during training but NOT in UI (safe defaults) ---
    if "center_distance_km" in X.columns:
        X["center_distance_km"] = 5.0  # median-like value

    if "distance_missing" in X.columns:
        X["distance_missing"] = 0

    if "is_dormitory" in X.columns:
        X["is_dormitory"] = 1 if room_type == "Dormitory" else 0

    if "multiple_bed_types" in X.columns:
        X["multiple_bed_types"] = 0

    # --- Manual One-Hot Encoding for Room Category ---
    room_mapping = {
        "Standard/Double": "Room_Category_Standard",
        "Deluxe/Superior": "Room_Category_Deluxe",
        "Premium/Executive": "Room_Category_Premium",
        "Family/Triple": "Room_Category_Family",
        "Suite/Luxury": "Room_Category_Suite",
        "Dormitory": "Room_Category_Dormitory"
    }

    mapped_col = room_mapping.get(room_type)
    if mapped_col in X.columns:
        X[mapped_col] = 1

    # --- Predict ---
    prediction = model.predict(X)[0]
    return prediction

# --------------------------------------------------
# Output Section
# --------------------------------------------------
if st.sidebar.button("Predict Hotel Price"):
    price = predict_price()

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="Estimated Price (₹)",
            value=f"₹{price:,.0f}"
        )

    with col2:
        st.info(
            """
            **How this prediction works**
            - Uses Gradient Boosting (no feature scaling)
            - Trained on scraped Delhi hotel listings
            - Log-transformed bed counts
            - District & transport encoded from training data
            """
        )
