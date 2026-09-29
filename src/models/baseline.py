import numpy as np

class NaiveBaselineModel:
    def __init__(self, method='drift'):
        self.method = method
        self.historical_returns = None
        
    def train_and_predict(self, train_df, test_df):
        """
        Train and predict the expected return to next Diwali.
        """
        if self.method == 'drift':
            # Average daily return over the training set
            daily_drift = np.mean(train_df['Target_Ret_1D'].dropna())
            days_to_diwali = test_df['Days_to_Diwali'].values[0]
            expected_return = (1 + daily_drift) ** days_to_diwali - 1
            return expected_return
        elif self.method == 'historical_average':
            # Just predict the historical average return to Diwali from T-30
            # Filter train_df for days around T-30
            t_30 = train_df[(train_df['Days_to_Diwali'] >= 25) & (train_df['Days_to_Diwali'] <= 35)]
            if not t_30.empty:
                return np.mean(t_30['Target_Return_to_Diwali'])
            return 0.0
        return 0.0
