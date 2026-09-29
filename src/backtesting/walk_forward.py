import pandas as pd
import numpy as np
import logging
from src.features.seasonality import DIWALI_DATES
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def generate_target_variable(df):
    """
    Generate the target variable: the price of the ETF on the next Diwali date.
    """
    df = df.copy()
    
    # We want to know the ETF price at the next Diwali.
    # We will map each Diwali date to its actual closest trading day price.
    diwali_prices = {}
    for d_date in DIWALI_DATES:
        # Find the closest trading day on or before Diwali
        available_dates = df.index[df.index <= d_date]
        if not available_dates.empty:
            closest_date = available_dates[-1]
            diwali_prices[d_date] = df.loc[closest_date, 'Target_Price']
            
    def get_next_diwali_price(current_date):
        future_diwalis = [d for d in DIWALI_DATES if d >= current_date]
        if not future_diwalis:
            return np.nan
        next_d = future_diwalis[0]
        return diwali_prices.get(next_d, np.nan)
        
    df['Next_Diwali_Price'] = df.index.to_series().apply(get_next_diwali_price)
    
    # Target return = (Next_Diwali_Price / Current_Price) - 1
    df['Target_Return_to_Diwali'] = (df['Next_Diwali_Price'] / df['Target_Price']) - 1
    
    return df

def walk_forward_backtest(df, model, start_year=2018):
    """
    Train model on data up to Year T, predict Diwali T+1.
    """
    results = []
    
    years = [y for y in range(start_year, 2026)]
    
    for test_year in years:
        if test_year == 2020:
            continue
            
        # Test Diwali is the Diwali of test_year
        test_diwali = [d for d in DIWALI_DATES if d.year == test_year][0]
        
        # We need to predict the price for test_diwali.
        # We can make a prediction 30 days before test_diwali.
        predict_date = test_diwali - pd.Timedelta(days=30)
        
        # Train data is everything before predict_date where Next_Diwali_Price is known
        # But wait, to avoid leakage, the target "Next_Diwali_Price" for the training set must be in the past!
        # So we only use rows where the 'Next_Diwali_Price' date is strictly before predict_date.
        
        train_df = df[df.index < predict_date].copy()
        # Drop rows where the target diwali is the test_diwali or later (data leakage!)
        train_df = train_df[train_df['Days_to_Diwali'] == train_df['Days_to_Diwali']] # Drop NA
        # Only keep train data where the next diwali was before our prediction date
        
        valid_train_dates = []
        for idx, row in train_df.iterrows():
            if idx.year == 2020:
                continue
            future_diwalis = [d for d in DIWALI_DATES if d >= idx]
            if future_diwalis and future_diwalis[0] < predict_date:
                valid_train_dates.append(idx)
                
        train_df = train_df.loc[valid_train_dates]
        
        if train_df.empty:
            continue
            
        # Get the row for prediction
        # Find the closest trading day to predict_date
        available_predict_dates = df.index[df.index <= predict_date]
        if available_predict_dates.empty:
            continue
            
        pred_idx = available_predict_dates[-1]
        test_row = df.loc[[pred_idx]].copy()
        
        actual_price = df.loc[pred_idx, 'Target_Price']
        actual_target_diwali_price = test_row['Next_Diwali_Price'].values[0]
        
        if pd.isna(actual_target_diwali_price):
            continue
            
        pred_return = model.train_and_predict(train_df, test_row)
        pred_diwali_price = actual_price * (1 + pred_return)
        
        results.append({
            'Test_Year': test_year,
            'Prediction_Date': pred_idx,
            'Current_Price': actual_price,
            'Actual_Diwali_Price': actual_target_diwali_price,
            'Predicted_Diwali_Price': pred_diwali_price
        })
        
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['Error'] = results_df['Predicted_Diwali_Price'] - results_df['Actual_Diwali_Price']
        results_df['Abs_Error'] = results_df['Error'].abs()
        results_df['Direction_Actual'] = np.sign(results_df['Actual_Diwali_Price'] - results_df['Current_Price'])
        results_df['Direction_Pred'] = np.sign(results_df['Predicted_Diwali_Price'] - results_df['Current_Price'])
        results_df['Direction_Correct'] = (results_df['Direction_Actual'] == results_df['Direction_Pred']).astype(int)
        
    return results_df
