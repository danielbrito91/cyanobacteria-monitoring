import argparse
from typing import Text

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data import preprocess
from src.utils.logs import get_logger


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


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict_cyano(config_path=args.config)
