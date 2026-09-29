import pandas as pd
import numpy as np

DIWALI_DATES = [
    pd.to_datetime('2015-11-11'),
    pd.to_datetime('2016-10-30'),
    pd.to_datetime('2017-10-19'),
    pd.to_datetime('2018-11-07'),
    pd.to_datetime('2019-10-27'),
    pd.to_datetime('2020-11-14'),
    pd.to_datetime('2021-11-04'),
    pd.to_datetime('2022-10-24'),
    pd.to_datetime('2023-11-12'),
    pd.to_datetime('2024-11-01'),
    pd.to_datetime('2025-10-21'),
    pd.to_datetime('2026-11-08')
]

def add_seasonality_features(df):
    """
    Calculate seasonality features like days to next Diwali.
    """
    df = df.copy()
    
    # Days to next Diwali
    def days_to_next_diwali(current_date):
        future_diwalis = [d for d in DIWALI_DATES if d >= current_date]
        if not future_diwalis:
            return np.nan
        return (future_diwalis[0] - current_date).days

    df['Days_to_Diwali'] = df.index.to_series().apply(days_to_next_diwali)
    
    # Is it within 30 days of Diwali?
    df['Is_Pre_Diwali_30D'] = (df['Days_to_Diwali'] <= 30) & (df['Days_to_Diwali'] >= 0)
    df['Is_Pre_Diwali_30D'] = df['Is_Pre_Diwali_30D'].astype(int)
    
    return df
