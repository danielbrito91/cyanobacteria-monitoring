import pandas as pd
import yaml
from typing import Text
import joblib
import argparse

from src.utils.logs import get_logger
from src.train.train_model import train_model
from src.features.featurize import create_ratios

def train(config_path: Text):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])
    
    logger.info("Load train and test set")
    df_train = pd.read_csv(config["data_split"]["trainset_path"])

    logger.info("Select features")
    selected_cols = config["featurize"]["selected_features"]
    target = config["featurize"]["target_column"]
    df_train = create_ratios(df_train)
    df_train = df_train[selected_cols + [target]]

    logger.info("Train selected regressor")
    selected_regressor = config["train"]["estimator_name"]
    trained_model = train_model(
        df = df_train,
        target_column = target,
        estimator_name = selected_regressor,
        params = config["train"]["estimators"][selected_regressor]["params"],
        polynomial_degree = config["featurize"]["poly_degree"]
        )
    
    logger.info("Save model")
    model_path = config["train"]["model_path"]
    joblib.dump(trained_model, model_path)
    
if __name__ == "__main__":
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)