import argparse
from typing import Text

import joblib
import json
import pandas as pd
import yaml
from src.data import preprocess

from src.train import train_model
from src.utils import logs


# Read data
def train_on_full_dataset(config_path: Text) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = logs.get_logger("TRAIN_FULL", log_level=config["base"]["log_level"])

    logger.info("Load full dataset")
    selected_cols = config["featurize"]["selected_features"]
    target = config["featurize"]["target_column"]
    full_df = pd.read_csv(config["data_load"]["labeled_df"])

    logger.info("Preprocess full dataset")
    full_df = preprocess.clean_data(full_df)
    full_df = preprocess.create_ratios(full_df)

    # id = full_df[["date", "Data da coleta"]]
    X_full = full_df[selected_cols]
    y_full = full_df[target]

    logger.info("Get the best regressor")
    # Keys: "model", "performance", "pred_true_plot"
    best_regressor_artifacts = train_model.get_best_model(
        config["mlflow_config"]["experiment_name"]
    )

    logger.info("Train selected regressor on full dataset")
    trained_model = best_regressor_artifacts["model"].fit(X_full, y_full)

    logger.info("Save model")
    joblib.dump(trained_model, config["train"]["full_model_path"])

    logger.info("Save other artifacts")
    train_model.save_dict(best_regressor_artifacts["performance"], config["evaluate"]["metrics_file"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_on_full_dataset(config_path=args.config)
