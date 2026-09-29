import pandas as pd
import logging
import yaml
from pathlib import Path
from .yahoo import fetch_yahoo_data
from .fred import fetch_fred_data

logger = logging.getLogger(__name__)

def load_all_data(config_path="config.yaml", force_refresh=False):
    """
    Load all configured data sources and merge them into a single dataframe.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    start_date = config['start_date']
    cache_dir = Path(config['paths']['data_cache'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Target ETF
    target_ticker = config['target_etf']
    df_target = fetch_yahoo_data(target_ticker, start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_target.empty:
        df_target = df_target[['Close', 'Volume']].rename(columns={'Close': 'Target_Price', 'Volume': 'Target_Volume'})
    else:
        raise ValueError(f"Could not load target ETF data for {target_ticker}")
        
    # 2. International Gold
    df_gold = fetch_yahoo_data(config['international_gold'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_gold.empty:
        df_gold = df_gold[['Close']].rename(columns={'Close': 'Gold_USD'})
        
    # 3. USD/INR
    df_usdinr = fetch_yahoo_data(config['usd_inr'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_usdinr.empty:
        df_usdinr = df_usdinr[['Close']].rename(columns={'Close': 'USD_INR'})
        
    # 4. Silver
    df_silver = fetch_yahoo_data(config['silver'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_silver.empty:
        df_silver = df_silver[['Close']].rename(columns={'Close': 'Silver_USD'})
        
    # 5. VIX
    df_vix = fetch_yahoo_data(config['vix'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_vix.empty:
        df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX'})
        
    # 6. US 10Y Yield
    df_us10y = fetch_yahoo_data(config['us_10y_yield'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_us10y.empty:
        df_us10y = df_us10y[['Close']].rename(columns={'Close': 'US_10Y'})
        
    # 7. US Dollar Index
    df_dxy = fetch_yahoo_data(config['us_dollar_index'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_dxy.empty:
        df_dxy = df_dxy[['Close']].rename(columns={'Close': 'DXY'})

    # 8. Nifty 50
    df_nifty = fetch_yahoo_data(config['nifty_50'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_nifty.empty:
        df_nifty = df_nifty[['Close']].rename(columns={'Close': 'Nifty50'})
        
    # 9. Gold VIX
    df_gvz = fetch_yahoo_data(config['gold_vix'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_gvz.empty:
        df_gvz = df_gvz[['Close']].rename(columns={'Close': 'Gold_VIX'})
        
    # 10. S&P 500
    df_sp500 = fetch_yahoo_data(config['sp500'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_sp500.empty:
        df_sp500 = df_sp500[['Close']].rename(columns={'Close': 'SP500'})
        
    # 11. FRED Fed Funds
    df_fedfunds = fetch_fred_data(config['fred_fedfunds'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_fedfunds.empty:
        df_fedfunds = df_fedfunds.rename(columns={'FEDFUNDS': 'Fed_Funds'})
        
    # 12. FRED 10Y Breakeven Inflation
    df_inflation = fetch_fred_data(config['fred_inflation_10y'], start_date, cache_dir=cache_dir, force_refresh=force_refresh)
    if not df_inflation.empty:
        df_inflation = df_inflation.rename(columns={'T10YIE': 'US_Inflation_10Y'})

    # Merge all DataFrames
    dfs = [df_target, df_gold, df_usdinr, df_silver, df_vix, df_us10y, df_dxy, df_nifty, df_gvz, df_sp500, df_fedfunds, df_inflation]
    dfs = [df for df in dfs if not df.empty]
    
    # Merge on date index
    if len(dfs) > 1:
        merged_df = dfs[0].join(dfs[1:], how='outer')
    else:
        merged_df = dfs[0]
        
    # Forward fill missing values up to 5 days generally, but fully forward fill monthly macro
    # Fully forward and back fill macro data
    merged_df = merged_df.ffill().bfill()
    
    # But drop rows where the original Target_Price was NaN (rely on actual trading days for the ETF)
    # Actually, ffill().bfill() filled Target_Price too. 
    # Let's drop weekends/holidays by intersecting with the original Target_Price index
    merged_df = merged_df.loc[df_target.index.intersection(merged_df.index)]
    
    # Remove egregious Yahoo Finance data errors (e.g., Target_Price < 10 when it should be ~30-40)
    merged_df = merged_df[merged_df['Target_Price'] >= 10.0]
    
    # Save processed raw data
    raw_path = Path(config['paths']['data_raw']) / "merged_raw.csv"
    merged_df.to_csv(raw_path)
    logger.info(f"Merged raw data saved to {raw_path}")
    
    return merged_df
