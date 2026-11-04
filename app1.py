#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model1.pkl")
scaler = joblib.load("scaler1.pkl")

st.set_page_config(page_title="insurance", page_icon="🛡️")

st.title("🛡️ insurance")

st.write("Enter the feature values below to predict the charges.")

# User Inputs
age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, format="%.3f")
children = st.selectbox("Children", [0, 1, 2, 3])
smoker = st.selectbox("Smoker", [0, 1])
region = st.selectbox("Region", [0, 1, 2, 3])

if st.button("Predict charges"):

    input_data = np.array([[
        age,
        gender,
        bmi,
        children,
        smoker,
        region,
    ]])


    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display Result
    st.success(f"🟢 Predicted Charges: {prediction[0]:,.2f}")



# In[ ]:




