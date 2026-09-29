import pandas as pd

def add_gold_specific_features(df):
    """
    Calculate relationships specific to Gold, USD/INR, and Silver.
    """
    df = df.copy()
    
    if 'Gold_USD' in df.columns and 'USD_INR' in df.columns:
        # Approximate INR Gold Price (assuming 1 troy ounce = 31.1035 grams)
        # International gold is per troy ounce in USD.
        # So INR per gram = (Gold_USD * USD_INR) / 31.1035
        # The ETF might track 1 gram or 1/100th of an ounce, we just look at relative movement.
        df['Approx_Gold_INR'] = df['Gold_USD'] * df['USD_INR']
        
        # Premium/Discount of ETF tracking vs Approx INR Gold
        # Normalizing to the first available ratio
        ratio = df['Target_Price'] / df['Approx_Gold_INR']
        baseline_ratio = ratio.dropna().iloc[0] if not ratio.dropna().empty else 1.0
        
        df['Tracking_Diff'] = ratio / baseline_ratio
        
        # Returns
        df['Gold_USD_Ret_1D'] = df['Gold_USD'].pct_change()
        df['Gold_USD_Ret_30D'] = df['Gold_USD'].pct_change(periods=30)
        df['Gold_USD_Ret_180D'] = df['Gold_USD'].pct_change(periods=180)
        df['USD_INR_Ret_1D'] = df['USD_INR'].pct_change()
        df['USD_INR_Ret_30D'] = df['USD_INR'].pct_change(periods=30)
        df['USD_INR_Ret_180D'] = df['USD_INR'].pct_change(periods=180)
        df['Approx_Gold_INR_Ret_1D'] = df['Approx_Gold_INR'].pct_change()
        df['Approx_Gold_INR_Ret_30D'] = df['Approx_Gold_INR'].pct_change(periods=30)

    if 'Silver_USD' in df.columns and 'Gold_USD' in df.columns:
        # Gold-Silver Ratio
        df['Gold_Silver_Ratio'] = df['Gold_USD'] / df['Silver_USD']
        
    return df
