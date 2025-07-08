# 📈 NSE Stock Price Predictor AI
A machine learning app that forecasts **next-day stock prices** on the **NSE (India)** using **LSTM** neural networks.

Built with [Darts](https://github.com/unit8co/darts), [Streamlit](https://streamlit.io/), and [Yahoo Finance](https://finance.yahoo.com/).

## 🔧 Features
- ✅ Predicts next-day closing prices using LSTM
- ✅ Interactive Streamlit dashboard
- ✅ Backtesting engine with buy/sell signals
- ✅ Easily extensible to 6-month forecasts (coming soon)

## 🚀 Installation
```bash
git clone https://github.com/yourusername/nse-stock-price-predictor-ai.git
cd nse-stock-price-predictor-ai
pip install -r requirements.txt
```

## 🧠 Usage
### ▶️ 1. Run the Dashboard
```bash
streamlit run app/nse_forecast_dashboard.py
```

### ▶️ 2. Run Backtest
```bash
python core/backtest_short_term.py
```

## ✅ Next Features (WIP)
- [ ] Long-term 6-month prediction (PatchTST or Prophet)
- [ ] Strategy optimizer
- [ ] Live alerts (Telegram/Email)
- [ ] Automated weekly retraining

## 📚 Dependencies
- `darts[torch]`
- `streamlit`
- `yfinance`
- `pandas`
- `scikit-learn`
- `matplotlib`

## 👨‍💻 Author
**PJV Financial**
