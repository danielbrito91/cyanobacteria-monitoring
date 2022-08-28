import pandas as pd
import yaml
from typing import Text
import joblib
import argparse
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np

from src.utils.logs import get_logger
from src.features.featurize import create_ratios, clean_and_select_data

def predict_cyano(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    logger = get_logger("PREDICT", log_level=config["base"]["log_level"])

    logger.info("Load model")
    model_path = config["train"]["full_model_path"]
    model = joblib.load(model_path)

    logger.info("Load S2A dataset")
    gee = pd.read_csv(config["data_load"]["s2a_df"])

    gee = clean_and_select_data(gee, config["featurize"]["selected_clean_columns"])
    gee = create_ratios(gee)
        
    id = gee["date"]

    selected_cols = config["featurize"]["selected_features"]
    X = gee[selected_cols]

    logger.info("Predict")
    y_pred = model.predict(X)
    y_pred = np.where(y_pred < 0, 0, y_pred)

    df_predicted = pd.DataFrame({
        "date": id,
        "y_pred": y_pred
    })

    logger.info("Save predicted data")
    df_predicted.to_csv(config["evaluate"]["final_predictions_file"], index=False)

if __name__ == "__main__":
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict_cyano(config_path=args.config)