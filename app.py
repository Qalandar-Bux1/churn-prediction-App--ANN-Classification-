import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and encoders/scaler
model = load_model("model.keras")  # <-- make sure matches how you saved

with open("onehot_encoder_geo", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("🏦 Bank Customer Churn Prediction (ANN)")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Prepare the input data
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# OneHot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# Combine all features
input_df = pd.concat([input_df, geo_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = float(prediction[0][0])

    if probability > 0.5:
        st.error(f"🚨 Customer will exit (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer will stay (Probability: {probability:.2f})")
