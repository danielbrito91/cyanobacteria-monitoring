from datetime import datetime

import numpy as np
import pandas as pd


def load_data(config):
    nome_eta = config["data_create"]["nome_eta"]

    gee = pd.read_csv(config["data_load"]["s2a_df"])
    gee["date"] = pd.to_datetime(gee["date"]).dt.date

    vigi = pd.read_csv(config["data_load"]["labels_df"])

    ciano_vigi = vigi.loc[
        (vigi["Parâmetro"] == "Cianobactérias")
        & (vigi["Unidade"] == "Total de cianobactérias")
        & (vigi["Nome da ETA / UTA"] == nome_eta),
        ["Data da coleta", "Resultado"],
    ]

    ciano_labels = ciano_vigi.loc[
        pd.to_datetime(ciano_vigi["Data da coleta"]).dt.date >= min(gee["date"])
    ]
    ciano_labels.loc[:, "Data da coleta"] = pd.to_datetime(
        ciano_labels["Data da coleta"]
    ).dt.date
    ciano_labels.loc[:, "Resultado"] = pd.to_numeric(ciano_labels["Resultado"])
    ciano_labels = ciano_labels.sort_values(by=["Data da coleta"])

    return gee, ciano_labels


def get_intervals(date_column, first_date, config):

    delta_d = config["data_create"]["delta_dias"]

    j_inicio = pd.date_range(
        first_date - pd.Timedelta(days=delta_d),
        datetime.today() - pd.Timedelta(days=delta_d),
        freq=str(delta_d * 2 + 1) + "D",
    )

    j_final = pd.date_range(
        first_date + pd.Timedelta(days=delta_d),
        datetime.today() + pd.Timedelta(days=delta_d),
        freq=str(delta_d * 2 + 1) + "D",
    )

    interval_df = pd.DataFrame(
        {"inicio": j_inicio, "fim": j_final, "interval": np.arange(len(j_final))}
    )

    intervals = []

    for date_i in date_column:
        interval_i = interval_df.loc[
            (pd.to_datetime(interval_df["inicio"]) <= pd.to_datetime(date_i))
            & (pd.to_datetime(interval_df["fim"]) >= pd.to_datetime(date_i)),
            "interval",
        ]

        if len(interval_i) > 0:
            interval_i = interval_i.values[0]
        else:
            print(date_i)

        intervals.append(interval_i)

    return intervals
