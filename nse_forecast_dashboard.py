import streamlit as st
import yfinance as yf
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Price Predictor", layout="wide")

def fetch_series(ticker):
    df = yf.download(ticker, start="2020-01-01")
    df = df[['Close']].dropna()
    return TimeSeries.from_dataframe(df), df

def predict_next_day(series):
    scaler = Scaler(StandardScaler())
    series_scaled = scaler.fit_transform(series)
    train, val = series_scaled.split_before(0.9)

    model = RNNModel(
        model="LSTM",
        input_chunk_length=60,
        output_chunk_length=1,
        n_epochs=30,
        random_state=42,
        verbose=False
    )
    model.fit(train, val_series=val)
    forecast = model.predict(1)
    forecast = scaler.inverse_transform(forecast)
    return forecast.values().flatten()[0]

# UI
st.title("📈 NSE Stock Price Predictor (Next-Day)")

ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE.NS, TCS.NS):", value="RELIANCE.NS")
if st.button("Predict"):
    with st.spinner("Fetching data and predicting..."):
        ts, raw_df = fetch_series(ticker)
        pred_price = predict_next_day(ts)
        last_price = raw_df['Close'].iloc[-1]

        st.metric("📌 Latest Close", f"₹{last_price:.2f}")
        st.metric("📉 Predicted Next-Day Close", f"₹{pred_price:.2f}")

        st.line_chart(raw_df['Close'][-180:])
