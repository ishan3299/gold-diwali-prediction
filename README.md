# Gold ETF Prediction System for Diwali 2026

## Objective
Predict the expected Indian Gold ETF price (and statistically plausible range) around Diwali 2026 (November 8, 2026).

## Pipeline
- **Data Engineering:** Automated fetching from Yahoo Finance (and placeholders for FRED/India local data) for Target ETF, International Gold, USD/INR, Silver, VIX, US 10Y Yield, Dollar Index, and Nifty 50.
- **Feature Engineering:** Technical indicators (moving averages, EMA, rolling volatility), Gold specific features (approximate INR gold price, tracking difference), Macro indicators, and Seasonality (days to next Diwali).
- **Forecasting Models:** Walk-forward backtesting using an Ensemble Model (Drift Baseline, Historical Average Baseline, XGBoost, and Random Forest).
- **Simulations:** Monte Carlo simulations representing Geometric Brownian Motion pathways based on the forecasted return.
- **Output:** Outputs to console, Markdown report, and visualizations.

## Quickstart

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (Data Collection, Backtest, Forecast, Reporting)
python main.py

# Force refresh data
python main.py --refresh
```
