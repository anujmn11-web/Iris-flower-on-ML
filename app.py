import streamlit as st
import joblib
import numpy as np

# Deployement code
model = joblib.load("Iris_model.pkl")

# Page title
st.title("Iris flower on ML")

# Table import
sepal_length = st.number_input("sepal_length")
sepal_width = st.number_input("sepal_width")
petal_length = st.number_input("petal_length")
petal_width = st.number_input("petal_width")

# Prediction
if st.button("Predict"):
  input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]]).astype(np.float64)
  prediction = model.predict(input_data)
  st.write(prediction)
