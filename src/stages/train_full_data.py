
import pandas as pd
from typing import Text
import yaml
import joblib
import argparse

from src.features.featurize import create_ratios
from src.train.train_model import train_model
from src.utils.logs import get_logger

# Read data
def train_on_full_dataset(config_path: Text):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("TRAIN_FULL", log_level=config["base"]["log_level"])

    logger.info("Load full dataset")
    full_df = pd.read_csv(config["data_load"]["labeled_df"])
    selected_cols = config["featurize"]["selected_features"]
    target = config["featurize"]["target_column"]
    id = full_df[["date", "Data da coleta"]]

    logger.info("Featurize full dataset")
    full_df = create_ratios(full_df)
    full_df = full_df[selected_cols + [target]]

    logger.info("Train selected regressor on full dataset")
    selected_regressor = config["train"]["estimator_name"]
    trained_model = train_model(
        df = full_df,
        target_column = target,
        estimator_name = selected_regressor,
        params = config["train"]["estimators"][selected_regressor]["params"],
        polynomial_degree = config["featurize"]["poly_degree"]
        )
        
    logger.info("Save model")
    model_path = config["train"]["full_model_path"]
    joblib.dump(trained_model, model_path)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_on_full_dataset(config_path=args.config)