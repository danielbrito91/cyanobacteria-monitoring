from typing import Text

import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from src.data import label_gee


def plot_predicted_values(config_path: Text):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    sheet_pred = "https://docs.google.com/spreadsheets/d/1HL9PO6TMQRHW3Z641zERfDRrscGpgXUf6ErpMOEUVLc/edit#gid=0"
    sheet_vigi = "https://docs.google.com/spreadsheets/d/1HL9PO6TMQRHW3Z641zERfDRrscGpgXUf6ErpMOEUVLc/edit#gid=1074409459"

    url_pred = sheet_pred.replace('/edit#gid=', '/export?format=csv&gid=')
    url_vigi = sheet_vigi.replace('/edit#gid=', '/export?format=csv&gid=')

    pred = pd.read_csv(url_pred, encoding="latin1", decimal=",")
    ciano = pd.read_csv(url_vigi, encoding="latin1", decimal=",")

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
