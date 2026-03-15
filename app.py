import streamlit as st
import numpy as np
import pandas as pd
import keras
import pickle

st.title("Customer Churn Prediction")
st.write("App started successfully")

# -----------------------------
# Load Model and Preprocessors
# -----------------------------
@st.cache_resource
def load_resources():
    st.write("Loading model and encoders...")

    try:
        model = keras.models.load_model("model.h5", compile=False)
        st.write("Model loaded")

        with open("label_encoder_gender.pkl", "rb") as f:
            label_encoder_gender = pickle.load(f)
        st.write("Gender encoder loaded")

        with open("onehot_encoder_geo.pkl", "rb") as f:
            onehot_encoder_geo = pickle.load(f)
        st.write("Geography encoder loaded")

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        st.write("Scaler loaded")

        return model, label_encoder_gender, onehot_encoder_geo, scaler

    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()


model, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Customer Details")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)

age = st.slider("Age", 18, 92)

credit_score = st.number_input("Credit Score", value=600)
balance = st.number_input("Balance", value=0.0)
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)

has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "CreditScore":[credit_score],
        "Gender":[label_encoder_gender.transform([gender])[0]],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[num_of_products],
        "HasCrCard":[has_cr_card],
        "IsActiveMember":[is_active_member],
        "EstimatedSalary":[estimated_salary]
    })

    # Encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine encoded geography with other inputs
    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    st.subheader("Prediction Result")
    st.write("Churn Probability:", prediction_probability)

    if prediction_probability > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
