import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("petrol price prediction")

# Read CSV
df = pd.read_csv("petrol.csv")

st.subheader("Petrol data")
st.dataframe(df)

X = df[["City_Code","Fuel_Code","Quantity"]]
Y = df["Price"]

# Model
model = LinearRegression()
model.fit(X, Y)

# Input
City_C = st.number_input("Enter City_C:", 1, 50)
Fuel_C = st.number_input("Enter Fuel_C:", 1, 50)
Qty = st.number_input("Enter Qty (in litres):", 1, 50)

# Prediction
pred = model.predict([[City_C, Fuel_C, Qty]])

st.subheader("Predicted Price")
st.write(int(pred[0]))
