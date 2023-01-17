from typing import Text

import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from src.data import label_gee


def plot_predicted_values(config_path: Text):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    pred = pd.read_csv(config["evaluate"]["final_predictions_file"])

    _, ciano = label_gee.load_data(config)

    gee_plot = go.Scatter(x=pred["date"], y=pred["y_pred"], name="Predicted values", mode="markers")

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
