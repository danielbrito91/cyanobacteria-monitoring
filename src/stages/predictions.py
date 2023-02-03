import argparse
from typing import Text

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data import preprocess, label_gee
from src.utils.logs import get_logger

import gspread as gs
from gspread_dataframe import set_with_dataframe

def predict_cyano(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("PREDICT", log_level=config["base"]["log_level"])

    logger.info("Load model")
    model_path = config["train"]["full_model_path"]
    model = joblib.load(model_path)

    logger.info("Load S2A dataset")
    gee = pd.read_csv(config["gee_clean"]["clean_data_path"])

    logger.info("Create features")
    gee_fts = preprocess.create_ratios(gee)
    gee_fts["delta_days"] = 0
    gee_fts = preprocess.create_poly_features(gee_fts, config, labeled=False)

    id = gee_fts["date"]

    selected_cols = config["featurize"]["selected_features"]
    X = gee_fts[selected_cols]

    logger.info("Predict")
    y_pred = model.predict(X)
    y_pred = np.where(y_pred < 0, 0, y_pred)

    df_predicted = pd.DataFrame({"date": id, "y_pred": y_pred})

    logger.info("Save predicted data")
    df_predicted.to_csv(config["evaluate"]["final_predictions_file"], index=False)

    logger.info("Export predictions to Google Sheets")

    gc = gs.service_account(filename="config/service-account.json")
    _, ciano = label_gee.load_data(config)

    sh_pred = gc.open_by_url("https://docs.google.com/spreadsheets/d/1HL9PO6TMQRHW3Z641zERfDRrscGpgXUf6ErpMOEUVLc/edit#gid=0")
    sh_ciano = gc.open_by_url("https://docs.google.com/spreadsheets/d/1HL9PO6TMQRHW3Z641zERfDRrscGpgXUf6ErpMOEUVLc/edit#gid=1074409459")
    
    ws_pred = sh.worksheet("previsto")
    ws_ciano = sh.worksheet("vigi")

    ws_pred.clear()
    ws_ciano.clear()

    set_with_dataframe(
        worksheet=ws_pred,
        dataframe=df_predicted,
        include_index=False,
        include_column_header=True,
        resize=True)
    
    set_with_dataframe(
        worksheet=ws_ciano,
        dataframe=ciano,
        include_index=False,
        include_column_header=True,
        resize=True)

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict_cyano(config_path=args.config)