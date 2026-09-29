import pandas as pd

def add_macro_features(df):
    """
    Calculate macro relationships.
    """
    df = df.copy()
    
    if 'VIX' in df.columns:
        df['VIX_Ret_1D'] = df['VIX'].pct_change()
        
    if 'US_10Y' in df.columns:
        df['US_10Y_Diff_1D'] = df['US_10Y'].diff()
        
    if 'DXY' in df.columns:
        df['DXY_Ret_1D'] = df['DXY'].pct_change()
        df['DXY_Ret_30D'] = df['DXY'].pct_change(periods=30)
        df['DXY_Ret_180D'] = df['DXY'].pct_change(periods=180)
        
    if 'Nifty50' in df.columns:
        df['Nifty50_Ret_1D'] = df['Nifty50'].pct_change()
        df['Nifty50_Ret_30D'] = df['Nifty50'].pct_change(periods=30)
        df['Nifty50_Ret_180D'] = df['Nifty50'].pct_change(periods=180)
        
    if 'Gold_VIX' in df.columns:
        df['Gold_VIX_Ret_1D'] = df['Gold_VIX'].pct_change()
        
    if 'SP500' in df.columns:
        df['SP500_Ret_1D'] = df['SP500'].pct_change()
        df['SP500_Ret_30D'] = df['SP500'].pct_change(periods=30)
        df['SP500_Ret_180D'] = df['SP500'].pct_change(periods=180)
        
    if 'Fed_Funds' in df.columns:
        df['Fed_Funds_Diff_1D'] = df['Fed_Funds'].diff()
        df['Fed_Funds_Diff_30D'] = df['Fed_Funds'].diff(periods=30)
        df['Fed_Funds_Diff_180D'] = df['Fed_Funds'].diff(periods=180)
        
    if 'US_Inflation_10Y' in df.columns:
        df['US_Inflation_10Y_Diff_1D'] = df['US_Inflation_10Y'].diff()
        df['US_Inflation_10Y_Diff_30D'] = df['US_Inflation_10Y'].diff(periods=30)
        df['US_Inflation_10Y_Diff_180D'] = df['US_Inflation_10Y'].diff(periods=180)
        
    return df
