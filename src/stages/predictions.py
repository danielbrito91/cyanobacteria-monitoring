import argparse
import os
from datetime import date
from typing import Text

import boto3
import joblib
import numpy as np
import pandas as pd
import s3fs
import yaml

from src.data import in_out, label_gee, preprocess
from src.utils.logs import get_logger


def get_last_prediction_path(config, fs):
    bucket_path = os.path.dirname(config["evaluate"]["final_predictions_file"].format(dt=date.today().strftime("%Y%m%d")))
    return "s3://" + np.sort([f for f in fs.ls(bucket_path) if "prediction" in f])[-1]

def predict_cyano(config_path: Text) -> None:
    fs = s3fs.S3FileSystem()
    s3_resource = boto3.resource("s3")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("PREDICT", log_level=config["base"]["log_level"])

    logger.info("Load model")
    model_path = config["train"]["full_model_path"]
    with fs.open(model_path, "rb") as f:
        model = joblib.load(f)

    logger.info("Load S2A dataset")
    gee = pd.read_parquet(fs.open(config["data_load"]["s2a_df"]))

    logger.info("Create features")
    gee_fts = (
        gee.pipe(preprocess.create_ratios)
        .assign(delta_days=0)
        .pipe(preprocess.create_poly_features, labeled=False, config=config)
    )

    id = gee_fts["date"]

    selected_cols = config["featurize"]["selected_features"]
    X = gee_fts[selected_cols]

    logger.info("Predict")
    y_pred = model.predict(X)
    y_pred = np.where(y_pred < 0, 0, y_pred)

    df_predicted = pd.DataFrame({"date": id, "y_pred": y_pred})

    logger.info("Save predicted data")
    today_ = date.today().strftime("%Y%m%d")
    s3_path = config["evaluate"]["final_predictions_file"].format(dt=today_)
    bucket, key = in_out.get_bucket_and_key_from_s3path(
        s3_path
    )
    in_out.save_file_in_s3(df_predicted, bucket, key, s3_resource)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict_cyano(config_path=args.config)
