import pandas as pd
import numpy as np

def add_technical_features(df, windows=[5, 10, 20, 30, 60, 90, 180, 365]):
    """
    Add technical features to the dataframe.
    """
    df = df.copy()
    
    # Target Returns
    df['Target_Ret_1D'] = df['Target_Price'].pct_change()
    df['Target_Log_Ret_1D'] = np.log(df['Target_Price'] / df['Target_Price'].shift(1))
    
    for w in windows:
        # Rolling Returns
        df[f'Target_Ret_{w}D'] = df['Target_Price'].pct_change(periods=w)
        
        # Volatility
        df[f'Target_Vol_{w}D'] = df['Target_Log_Ret_1D'].rolling(window=w).std() * np.sqrt(252)
        
        # Moving Averages
        df[f'Target_MA_{w}D'] = df['Target_Price'].rolling(window=w).mean()
        
        # EMA
        df[f'Target_EMA_{w}D'] = df['Target_Price'].ewm(span=w, adjust=False).mean()
        
        # Momentum (Price / MA)
        df[f'Target_Mom_{w}D'] = df['Target_Price'] / df[f'Target_MA_{w}D'] - 1
        
        # Rolling Max/Min (Resistance and Support)
        df[f'Target_Max_{w}D'] = df['Target_Price'].rolling(window=w).max()
        df[f'Target_Min_{w}D'] = df['Target_Price'].rolling(window=w).min()
        
        # Drawdown / Distance to Resistance
        df[f'Target_Dist_Resist_{w}D'] = df['Target_Price'] / df[f'Target_Max_{w}D'] - 1
        
        # Distance to Support
        df[f'Target_Dist_Support_{w}D'] = df['Target_Price'] / df[f'Target_Min_{w}D'] - 1
        
    return df
