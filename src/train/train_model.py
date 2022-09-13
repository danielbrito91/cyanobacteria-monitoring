import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from typing import Dict, Text
from xgboost import XGBRegressor
from urllib.parse import urlparse

import mlflow
import mlflow.xgboost
import mlflow.sklearn

class UnsupportedRegressor(Exception):

    def __init__(self, estimator_name):
        self.msg = f"Unsupported regressor {estimator_name}"
        super().__init__(self.msg)
    
def get_supported_estimator() -> Dict:
    return {
        'ridge': Ridge,
        'xgboost': XGBRegressor,
        'random_forest': RandomForestRegressor
    }

def train_model(df: pd.DataFrame, target_column: Text, 
    estimator_name: Text, params: Dict, polynomial_degree: int):

    estimators = get_supported_estimator()
    if estimator_name not in estimators.keys():
        raise UnsupportedRegressor(estimator_name)

    # Start MLFlow
    with mlflow.start_run():
        regressor = estimators[estimator_name](**params)
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        poly_reg = Pipeline([
                ("poly", poly),
                ("regressor", regressor)
            ])
        
        model = TransformedTargetRegressor(
            regressor = poly_reg,
            transformer=PowerTransformer(method="yeo-johnson")
        )

        # Get X and y
        X_train = df.drop(columns = [target_column])
        y_train = df[target_column]

        model.fit(X_train, y_train)

        for param, value in params.items():
            mlflow.log_param(param, value)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            if estimator_name !=  "xgboost":
                mlflow.sklearn.log_model(model, "model", registered_model_name=estimator_name)
            else:
                mlflow.xgboost.log_model(model, "model", registered_model_name=estimator_name)
        else:
            if estimator_name !=  "xgboost":
                mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.xgboost.log_model(model, "model")
   
    return model