import argparse
from typing import Text

import boto3
import pandas as pd
import s3fs
import yaml

from src.data import in_out, label_gee, preprocess
from src.utils.logs import get_logger


def create_features(config_path: Text):
    """Create features

    Args:
        config_path (Text): _description_
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    fs = s3fs.S3FileSystem()
    s3_resource = boto3.resource("s3")

    logger = get_logger("FEATURE_ENG", log_level=config["base"]["log_level"])

    # Load labeled data
    gee, labels = label_gee.load_data(config, fs)
    df = label_gee.join_gee_labels(gee, labels, config)

    # Create features and clean
    logger.info("Create features")
    df = (
        df.pipe(preprocess.create_ratios)
        .pipe(preprocess.create_delta_days)
        .pipe(preprocess.create_poly_features, config=config)
    )

    logger.info("Select features")
    df_selected = df[
        config["featurize"]["selected_features"]
        + ["date", "Data da coleta", "Resultado"]
    ]

    logger.info("Save data with selected features")
    bucket, key = in_out.get_bucket_and_key_from_s3path(
        config["featurize"]["ft_data_path"]
    )
    in_out.save_file_in_s3(df_selected, bucket, key, s3_resource)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    create_features(config_path=args.config)
