import argparse
from typing import Text

import joblib
import pandas as pd
import s3fs
import yaml
from sklearn.linear_model import PoissonRegressor

from src.utils import logs


# Read data
def train_on_full_dataset(config_path: Text) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    fs = s3fs.S3FileSystem()

    logger = logs.get_logger("TRAIN_FULL", log_level=config["base"]["log_level"])

    logger.info("Load full dataset")
    selected_cols = config["featurize"]["selected_features"]
    target = config["featurize"]["target_column"]

    full_df = pd.read_parquet(fs.open(config["featurize"]["ft_data_path"]))
    # id = full_df[["date", "Data da coleta"]]
    X_full = full_df[selected_cols]
    y_full = full_df[target]

    logger.info("Train selected regressor on full dataset")
    selected_model = config["train"]["estimator_name"]
    if selected_model == "poisson_reg":
        params = config["train"]["estimators"][selected_model]["params"]
        trained_model = PoissonRegressor(**params)

    trained_model.fit(X_full, y_full)

    logger.info("Save model")
    model_path = config["train"]["full_model_path"]
    fs = s3fs.S3FileSystem()
    with fs.open(model_path, "wb") as f:
        joblib.dump(trained_model, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_on_full_dataset(config_path=args.config)
