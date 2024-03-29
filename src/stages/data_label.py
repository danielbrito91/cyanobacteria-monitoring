import argparse
from typing import Text

import pandas as pd
import s3fs
import yaml

from src.data import label_gee, preprocess
from src.utils.logs import get_logger


def data_label(config_path: Text) -> None:
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("DATA_LABEL", log_level=config["base"]["log_level"])

    fs = s3fs.S3FileSystem()
    gee, ciano_labels = label_gee.load_data(config, fs)
    gee["interval"] = label_gee.get_intervals(gee["date"], min(gee["date"]), config)

    logger.info(
        f"Labeling the data ({config['data_create']['delta_dias']}-day interval)"
    )
    ciano_labels["interval"] = label_gee.get_intervals(
        ciano_labels["Data da coleta"], min(gee["date"]), config
    )
    df = pd.merge(gee, ciano_labels, on="interval")

    logger.info("Select columns")
    selected_columns = config["featurize"]["selected_clean_columns"] + [
        config["featurize"]["target_column"],
        "Data da coleta",
        "interval",
    ]
    labeled_df = df[selected_columns]

    logger.info("Save the labeled data")
    labeled_df.to_csv(config["data_load"]["labeled_df"], index=False)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_label(config_path=args.config)
