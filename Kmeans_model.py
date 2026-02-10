import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = pickle.load(open("Kmeans Clustering.joblib", "rb"))
scaler = pickle.load(open("scaler.joblib", "rb"))

st.title("🌍 Country Development Clustering App")

st.write("Enter country details to predict cluster")

# Input fields (edit names based on your dataset columns)
gdp = st.number_input("GDP")
co2 = st.number_input("CO2 Emissions")
energy = st.number_input("Energy Usage")
tourism = st.number_input("Tourism")

if st.button("Predict Cluster"):
    
    data = np.array([[gdp, co2, energy, tourism]])
    
    # Scale input
    data_scaled = scaler.transform(data)
    
    # Predict cluster
    cluster = model.predict(data_scaled)
    
    st.success(f"This country belongs to Cluster: {cluster[0]}")
