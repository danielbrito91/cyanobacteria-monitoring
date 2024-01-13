import os
from datetime import date
from typing import Text

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots


def get_last_prediction_path(config, fs):
    bucket_path = os.path.dirname(
        config["evaluate"]["final_predictions_file"].format(
            dt=date.today().strftime("%Y%m%d")
        )
    )
    return "s3://" + np.sort([f for f in fs.ls(bucket_path) if "prediction" in f])[-1]


def plot_predicted_values(pred: pd.DataFrame, ciano: pd.DataFrame, config: dict):

    gee_plot = go.Scatter(
        x=pred["date"], y=pred["y_pred"], name="Predicted values", mode="markers"
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
