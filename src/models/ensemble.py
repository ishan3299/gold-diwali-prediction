import numpy as np

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [0.1, 0.1, 0.4, 0.4]
        else:
            self.weights = weights
            
    def train_and_predict(self, train_df, test_df):
        preds = []
        for model in self.models:
            pred = model.train_and_predict(train_df, test_df)
            preds.append(pred)
            
        ensemble_pred = np.average(preds, weights=self.weights)
        return ensemble_pred
