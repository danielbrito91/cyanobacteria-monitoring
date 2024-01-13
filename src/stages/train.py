import argparse
import json
from typing import Text

import joblib
import pandas as pd
import s3fs
import yaml
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from src.data import preprocess
from src.utils.logs import get_logger


def train(config_path: Text):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    fs = s3fs.S3FileSystem()

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    # Load labeled data
    logger.info("Load data")
    df = pd.read_parquet(fs.open(config["featurize"]["ft_data_path"]))

    logger.info("Split")
    df_train, df_test = preprocess.train_test_split_ts(df, config)

    features = config["featurize"]["selected_features"]

    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train[config["featurize"]["target_column"]]
    y_test = df_test[config["featurize"]["target_column"]]

    logger.info("Train")
    selected_model = config["train"]["estimator_name"]
    if selected_model == "poisson_reg":
        params = config["train"]["estimators"][selected_model]["params"]
        model = PoissonRegressor(**params)

    model.fit(X_train, y_train)

    logger.info("Evaluate")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mlse = mean_squared_log_error(y_test, y_pred)

    with open("reports/metrics_file.json", "w") as f:
        json.dump({"mae": mae, "mlse": mlse}, f)

    # Write
    logger.info("Save model")
    model_path = config["train"]["model_path"]
    fs = s3fs.S3FileSystem()
    with fs.open(model_path, "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
