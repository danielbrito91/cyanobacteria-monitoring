import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Text
import yaml
import pandas as pd

from src.data.label_gee import load_data

def plot_predicted_values(config_path: Text):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    pred = pd.read_csv(config["evaluate"]["final_predictions_file"])

    _, ciano = load_data("params.yaml")

    gee_plot = go.Scatter(
        x=pred["date"],
        y=pred["y_pred"],
        name="Predicted values")

    vigi_plot = go.Scatter(
        x=ciano["Data da coleta"],
        y=ciano["Resultado"],
        mode="markers",
        name="SISAGUA")

    fig = make_subplots()
    fig.add_trace(gee_plot)
    fig.add_trace(vigi_plot)

    return fig.update_layout(
        template="plotly_white",
        title=f"{config['data_create']['manancial']} - WTP {config['data_create']['nome_eta']}<br>Cyanobacteria monitoring",
        xaxis_title="Date",
        yaxis_title="Cyanobacteria (cells mL−1)")