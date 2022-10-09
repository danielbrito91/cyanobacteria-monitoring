import json
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from dateutil.relativedelta import relativedelta
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from xgboost import XGBRegressor

from src.data import preprocess


class UnsupportedRegressor(Exception):
    def __init__(self, estimator_name):
        self.msg = f"Unsupported regressor {estimator_name}"
        super().__init__(self.msg)


def get_supported_estimator() -> Dict:
    return {
        "ridge": Ridge,
        "xgboost": XGBRegressor,
        "random_forest": RandomForestRegressor,
    }


def plot_predicted_vs_labels(df):

    max_value = max(max(df["y_pred"]), max(df["y_true"]))

    plot = px.scatter(
        df,
        x="y_pred",
        y="y_true",
        template="plotly_white",
        labels={"y_pred": "Predicted value", "y_true": "True value"},
        title="Cyanobacteria monitoring prediction with S2A",
    )

    return plot.update_layout(
        shapes=[{"type": "line", "y0": 0, "y1": max_value, "x0": 0, "x1": max_value}]
    )


def train_model(
    df: pd.DataFrame,  # labeled df after clean and feat
    config: dict,
):

    # Read splitted df
    df_train_val, df_test = pd.read_csv(
        config["data_split"]["trainset_path"]
    ), pd.read_csv(config["data_split"]["testset_path"])
    target_column = config["featurize"]["target_column"]
    selected_columns = config["featurize"]["selected_features"]

    id_test = df_test[["date", "Data da coleta"]]
    X_test = df_test[selected_columns]
    y_test = df_test[target_column]

    # Model
    estimator_name = config["train"]["estimator_name"]
    estimators = get_supported_estimator()
    if estimator_name not in estimators.keys():
        raise UnsupportedRegressor(estimator_name)
    params = config["train"]["estimators"][estimator_name]["params"]

    regressor = estimators[estimator_name](**params)
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=PowerTransformer(method="yeo-johnson")
    )

    # TS Cross-Validation - optuna
    """tscv = TimeSeriesSplit(n_splits=2)
    for train_index, val_index in tscv.split(df_train_val):
        df_train, df_val = df_train_val.iloc[train_index], df_train_val.iloc[val_index]

        # Oversampling training data
        df_over = preprocess.oversampling(df_train, config)
        X_over = df_over[selected_columns].drop(columns="date")
        y_over = df_over[target_column]

        X_val = df_val[selected_columns].drop(columns="date")
        y_val = df_val[target_column]

        model.fit(X_over, y_over)
        y_pred = model.predict(X_val)
        y_pred = np.where(y_pred < 0, 0, y_pred)

        # Validation eval
        print(mean_absolute_error(y_val, y_pred))
"""
    # Evaluation on test
    df_train_over = preprocess.oversampling(df_train_val, config)
    model.fit(
        df_train_over[selected_columns],
        df_train_over[target_column],
    )
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred < 0, 0, y_pred)

    # Plot eval
    df_plot = pd.DataFrame(
        {
            "date": id_test["date"],
            "amostragem": id_test["Data da coleta"],
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    predicted_vs_true_plot = plot_predicted_vs_labels(df_plot)

    # Metrics eval
    performance = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mlse": mean_squared_log_error(y_test, y_pred),
    }

    return {
        "model": model,
        # "tscv": tscv,
        "performance": performance,
        "params": params,
        "pred_vs_true_plot": predicted_vs_true_plot,
    }


def simple_heuristic_baseline():
    # Model with only one feature: the month of the measure
    pass


def save_dict(d, filepath):
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def load_dict(filepath):
    """Load a dict from a json file."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def get_best_model(experiment_name: str) -> dict:
    """Get the artifacts from the best model

    Args:
        experiment_name (str): name of MLFlow experiment

    Returns:
        dict: model (model.pkl object), "performance"
        (dict containing metrics of the best model)
    """
    # mlflow.set_tracking_uri("file:///" +  "mlruns")
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    experiment_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.mae"]
    )
    best_run_id = experiment_runs.iloc[0].run_id
    # best_run = mlflow.get_run(run_id=best_run_id)
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as dp:
        client.download_artifacts(run_id=best_run_id, path="", dst_path=dp)
        model = joblib.load(Path(dp, "model.pkl"))
        performance = load_dict(filepath=Path(dp, "performance.json"))

    return {"model": model, "performance": performance}
