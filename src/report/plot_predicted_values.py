import os
from datetime import date
from typing import Text

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.interpolate import Rbf


def get_last_prediction_path(config, fs):
    bucket_path = os.path.dirname(
        config["evaluate"]["final_predictions_file"].format(
            dt=date.today().strftime("%Y%m%d")
        )
    )
    return "s3://" + np.sort([f for f in fs.ls(bucket_path) if "prediction" in f])[-1]


def smooth_predicted_values(pred: pd.DataFrame, smooth: int = 2) -> pd.DataFrame:
    rbf = Rbf(pred["date"], pred["y_pred"], smooth=smooth)

    dias = pd.date_range(pred["date"].min(), pred["date"].max(), freq="1d")

    return pd.DataFrame(dict(date=dias, y=rbf(dias)))


def plot_predicted_values(pred: pd.DataFrame, ciano: pd.DataFrame, config: dict):
    smoothed_df = smooth_predicted_values(pred)
    gee_plot = go.Scatter(
        x=smoothed_df["date"], y=smoothed_df["y"], name="Predicted values (RBF smooth)", mode="lines"
    )

    vigi_plot = go.Scatter(
        x=ciano["Data da coleta"], y=ciano["Resultado"], mode="markers", name="SISAGUA"
    )

    fig = make_subplots()
    fig.add_trace(gee_plot)
    fig.add_trace(vigi_plot)

    return fig.update_layout(
        template="plotly_white",
        title=f"{config['data_create']['manancial']} - WTP {config['data_create']['nome_eta']}<br>Cyanobacteria monitoring",
        xaxis_title="Date",
        yaxis_title="Cyanobacteria (cells mL−1)",
    )
