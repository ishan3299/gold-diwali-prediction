import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data.loaders import load_all_data
from src.features.technical import add_technical_features
from src.features.gold_features import add_gold_specific_features
from src.features.seasonality import add_seasonality_features, DIWALI_DATES
from src.features.macro import add_macro_features
from src.backtesting.walk_forward import generate_target_variable, walk_forward_backtest
from src.models.baseline import NaiveBaselineModel
from src.models.ml import MLModel
from src.models.ensemble import EnsembleModel
from src.models.monte_carlo import run_monte_carlo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_charts(df, backtest_results, mc_sims, mc_results, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Indian Gold ETF historical price
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Target_Price'])
    plt.title('Indian Gold ETF Historical Price')
    plt.savefig(output_dir / '1_historical_price.png')
    plt.close()
    
    # 2. International gold vs Indian gold (Normalized)
    if 'Gold_USD' in df.columns:
        plt.figure(figsize=(10,6))
        d_norm = df[['Target_Price', 'Gold_USD']].dropna()
        if not d_norm.empty:
            (d_norm / d_norm.iloc[0]).plot(figsize=(10,6))
            plt.title('International Gold vs Indian Gold ETF (Normalized)')
            plt.savefig(output_dir / '2_gold_vs_inr_gold.png')
            plt.close()
            
    # 3. Gold vs USD/INR
    if 'USD_INR' in df.columns:
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax2 = ax1.twinx()
        ax1.plot(df.index, df['Target_Price'], 'g-')
        ax2.plot(df.index, df['USD_INR'], 'b-')
        ax1.set_ylabel('Gold ETF Price', color='g')
        ax2.set_ylabel('USD/INR', color='b')
        plt.title('Gold ETF vs USD/INR')
        plt.savefig(output_dir / '3_gold_vs_usdinr.png')
        plt.close()
        
    # 7. Monte Carlo Distribution
    if mc_sims is not None:
        plt.figure(figsize=(10,6))
        for i in range(min(100, mc_sims.shape[0])):
            plt.plot(mc_sims[i, :], color='grey', alpha=0.1)
        plt.axhline(mc_results['P50'], color='red', label='Median')
        plt.axhline(mc_results['P10'], color='blue', linestyle='--', label='P10')
        plt.axhline(mc_results['P90'], color='blue', linestyle='--', label='P90')
        plt.title('Monte Carlo Price Paths to Diwali 2026')
        plt.legend()
        plt.savefig(output_dir / '7_monte_carlo.png')
        plt.close()

def generate_report(config, df, backtest_results, mc_results, ensemble_pred, final_price, target_diwali_date, output_path):
    report = f"""# Diwali 2026 Gold ETF Forecast

## Executive Summary
* Target ETF: {config['target_etf']}
* Current price: ₹{final_price:.2f}
* Diwali target date: {target_diwali_date.strftime('%Y-%m-%d')}
* Point Forecast: ₹{final_price * (1 + ensemble_pred):.2f}

## Market Context
* Latest international gold: {df['Gold_USD'].dropna().iloc[-1]:.2f} USD
* Latest USD/INR: {df['USD_INR'].dropna().iloc[-1]:.2f}

## Model Results
**Backtest MAE:** {backtest_results['Abs_Error'].mean():.2f}
**Backtest RMSE:** {np.sqrt((backtest_results['Error']**2).mean()):.2f}
**Directional Accuracy:** {(backtest_results['Direction_Correct'].mean() * 100):.1f}%

## Ensemble Forecast
* Expected Price: ₹{final_price * (1 + ensemble_pred):.2f}
* P10: ₹{mc_results['P10']:.2f}
* P25: ₹{mc_results['P25']:.2f}
* P50: ₹{mc_results['P50']:.2f}
* P75: ₹{mc_results['P75']:.2f}
* P90: ₹{mc_results['P90']:.2f}

## Scenario Analysis
| Scenario | Estimated Range |
| -------- | --------------: |
| Bear     | <= ₹{mc_results['P25']:.2f} |
| Base     | ₹{mc_results['P25']:.2f} - ₹{mc_results['P75']:.2f} |
| Bull     | >= ₹{mc_results['P75']:.2f} |
| Tail     | >= ₹{mc_results['P90']:.2f} |

## Data Quality
Sources: Yahoo Finance (yfinance)
Last update: {df.index[-1].strftime('%Y-%m-%d')}
Observations: {len(df)}
"""
    with open(output_path, 'w') as f:
        f.write(report)
        
    print("========================================")
    print("GOLD ETF — DIWALI 2026 FORECAST")
    print("========================================")
    print(f"Target ETF: {config['target_etf']}")
    print(f"Target trading date: {target_diwali_date.strftime('%Y-%m-%d')}")
    print(f"Latest price: ₹{final_price:.2f}")
    print(f"Latest international gold: {df['Gold_USD'].dropna().iloc[-1]:.2f}")
    print(f"Latest USD/INR: {df['USD_INR'].dropna().iloc[-1]:.2f}")
    print("-" * 16)
    print("MODEL FORECAST")
    print("-" * 16)
    print(f"Point estimate: ₹{final_price * (1 + ensemble_pred):.2f}")
    print(f"P10: ₹{mc_results['P10']:.2f}")
    print(f"P25: ₹{mc_results['P25']:.2f}")
    print(f"P50: ₹{mc_results['P50']:.2f}")
    print(f"P75: ₹{mc_results['P75']:.2f}")
    print(f"P90: ₹{mc_results['P90']:.2f}")
    print("========================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--forecast', action='store_true')
    args = parser.parse_args()

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Loading Data...")
    df = load_all_data("config.yaml", force_refresh=args.refresh)
    
    logger.info("Feature Engineering...")
    df = add_technical_features(df, config['features']['rolling_windows'])
    df = add_gold_specific_features(df)
    df = add_seasonality_features(df)
    df = add_macro_features(df)
    
    logger.info("Generating Target...")
    df = generate_target_variable(df)
    
    models = [
        NaiveBaselineModel(method='drift'),
        NaiveBaselineModel(method='historical_average'),
        MLModel(model_type='xgboost'),
        MLModel(model_type='rf'),
        MLModel(model_type='ridge'),
        MLModel(model_type='hist_gb')
    ]
    # Update ensemble weights to include HistGB
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel(models, weights=[0.05, 0.05, 0.25, 0.25, 0.15, 0.25])
    
    # Backtesting
    backtest_results = pd.DataFrame()
    if args.backtest or args.forecast or not (args.backtest and args.forecast):
        logger.info("Running Walk-Forward Backtest...")
        backtest_results = walk_forward_backtest(df, ensemble, start_year=2018)
        if not backtest_results.empty:
            logger.info(f"Backtest MAE: {backtest_results['Abs_Error'].mean():.2f}")
            backtest_results.to_csv("data/raw/backtest_results.csv", index=False)
    
    if args.forecast or not args.backtest:
        logger.info("Generating Forecast for Diwali 2026...")
        target_diwali_date = pd.to_datetime(config['target_date'])
        
        current_date = df.index[-1]
        final_price = df['Target_Price'].iloc[-1]
        
        # We want to predict the return to next Diwali
        # Create a single row test_df for today
        test_df = df.iloc[[-1]].copy()
        days_to_target = (target_diwali_date - current_date).days
        test_df['Days_to_Diwali'] = days_to_target
        
        # Train on all valid data where target is known
        train_df = df.dropna(subset=['Target_Return_to_Diwali'])
        
        # Exclude 2020 COVID anomaly
        train_df = train_df[train_df.index.year != 2020]
        
        pred_return = ensemble.train_and_predict(train_df, test_df)
        
        # Monte Carlo
        annual_drift = pred_return * (365 / days_to_target)
        annual_vol = df['Target_Log_Ret_1D'].std() * np.sqrt(252)
        
        mc_results = run_monte_carlo(final_price, days_to_target, annual_drift, annual_vol, config['models']['monte_carlo_simulations'])
        
        logger.info("Generating Outputs...")
        generate_charts(df, backtest_results, mc_results['simulations'], mc_results, config['paths']['outputs_charts'])
        generate_report(config, df, backtest_results, mc_results, pred_return, final_price, target_diwali_date, Path(config['paths']['outputs_reports']) / 'diwali_2026_gold_forecast.md')
        
if __name__ == "__main__":
    main()
