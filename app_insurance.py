from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model("insurance_dt_model")

input_direct = {"age":20, "sex":"male", "bmi":20, "children":2, "smoker":"yes", "region":"southwest"}
input_df = pd.DataFrame([input_direct])
predictions_df = predict_model(estimator=model, data=input_df)

print(predictions_df)