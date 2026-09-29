import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel

class MLModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        if model_type == 'xgboost':
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif model_type == 'ridge':
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif model_type == 'hist_gb':
            self.model = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
            
    def _get_features(self, df):
        # Select features to use
        features = [
            'Days_to_Diwali', 'Target_Ret_30D', 'Target_Ret_90D', 'Target_Vol_30D', 'Target_Mom_30D', 
            'Gold_USD_Ret_30D', 'USD_INR_Ret_30D', 'VIX', 'Gold_VIX',
            'SP500_Ret_30D', 'Nifty50_Ret_30D', 'DXY_Ret_30D', 'US_10Y', 'Fed_Funds', 'US_Inflation_10Y',
            'US_Inflation_10Y_Diff_30D', 'Fed_Funds_Diff_30D',
            'Gold_USD_Ret_180D', 'USD_INR_Ret_180D', 'SP500_Ret_180D', 'Nifty50_Ret_180D', 'DXY_Ret_180D',
            'US_Inflation_10Y_Diff_180D', 'Fed_Funds_Diff_180D',
            'Target_Dist_Resist_10D', 'Target_Dist_Support_10D',
            'Target_Dist_Resist_30D', 'Target_Dist_Support_30D',
            'Target_Dist_Resist_90D', 'Target_Dist_Support_90D',
            'Target_Dist_Resist_365D', 'Target_Dist_Support_365D'
        ]
        # Keep only features that exist in df
        features = [f for f in features if f in df.columns]
        return features

    def train_and_predict(self, train_df, test_df):
        features = self._get_features(train_df)
        
        X_train = train_df[features].fillna(0) # Need to fillna for Ridge and RF
        y_train = train_df.loc[X_train.index, 'Target_Return_to_Diwali']
        
        # Use a base Random Forest to select the top most important features
        # This prevents the final models from overfitting to noise
        if self.model_type != 'ridge':
            selector = SelectFromModel(RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42), prefit=False)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            
            X_test = test_df[features].fillna(0)
            X_test = selector.transform(X_test)
        else:
            X_test = test_df[features].fillna(0)
        
        self.model.fit(X_train, y_train)
        pred_return = self.model.predict(X_test)[0]
        return pred_return
