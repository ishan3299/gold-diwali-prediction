import pandas as pd
import numpy as np
from src.features.seasonality import DIWALI_DATES
import matplotlib.pyplot as plt

def main():
    # Load daily prices
    df = pd.read_csv("data/raw/merged_raw.csv", index_col=0, parse_dates=True)
    prices = df['Target_Price'].dropna()
    
    results = []
    
    # We want past diwali dates strictly in the past (up to 2025)
    past_diwalis = [d for d in DIWALI_DATES if d <= prices.index[-1]]
    
    for d_date in past_diwalis:
        # Get the actual trading date on or just before Diwali
        d_idx = prices.index[prices.index <= d_date]
        if d_idx.empty:
            continue
        t0 = d_idx[-1]
        
        # Helper to get price N calendar days offset
        def get_price(offset_days):
            target_date = d_date + pd.Timedelta(days=offset_days)
            idx = prices.index[prices.index <= target_date]
            if idx.empty:
                return np.nan
            # We want the closest trading day to the target date. If offset is negative, it should be on or before.
            # If positive, it should be on or before.
            return prices.loc[idx[-1]]
            
        p_t30 = get_price(-30)
        p_t15 = get_price(-15)
        p_t7 = get_price(-7)
        p_t0 = prices.loc[t0]
        p_t7_post = get_price(7)
        p_t15_post = get_price(15)
        p_t30_post = get_price(30)
        
        results.append({
            'Year': d_date.year,
            'Diwali_Date': d_date.strftime('%Y-%m-%d'),
            'T-30 to T0 (%)': (p_t0 / p_t30 - 1) * 100 if p_t30 else np.nan,
            'T-15 to T0 (%)': (p_t0 / p_t15 - 1) * 100 if p_t15 else np.nan,
            'T-7 to T0 (%)': (p_t0 / p_t7 - 1) * 100 if p_t7 else np.nan,
            'T0 to T+7 (%)': (p_t7_post / p_t0 - 1) * 100 if p_t7_post else np.nan,
            'T0 to T+15 (%)': (p_t15_post / p_t0 - 1) * 100 if p_t15_post else np.nan,
            'T0 to T+30 (%)': (p_t30_post / p_t0 - 1) * 100 if p_t30_post else np.nan,
        })
        
    res_df = pd.DataFrame(results)
    print("=== Diwali Gold Price Action (Last 10 Years) ===")
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x) if pd.notnull(x) else "NaN"))
    
    print("\n=== Summary Statistics ===")
    numeric_cols = [c for c in res_df.columns if '%' in c]
    summary = res_df[numeric_cols].agg(['mean', 'median', 'std', lambda x: (x > 0).mean() * 100]).rename(index={'<lambda>': 'Win Rate (%)'})
    print(summary.to_string(float_format=lambda x: "{:.2f}".format(x) if pd.notnull(x) else "NaN"))
    
    # Save a chart
    res_df.set_index('Year')[numeric_cols[:3]].plot(kind='bar', figsize=(10, 6))
    plt.title("Pre-Diwali Gold ETF Returns (Last 10 Years)")
    plt.ylabel("Return (%)")
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig('outputs/charts/diwali_seasonality.png')

if __name__ == "__main__":
    main()
