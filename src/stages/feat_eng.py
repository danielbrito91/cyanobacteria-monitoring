import argparse
from typing import Text

import pandas as pd
import yaml

from src.data import preprocess
from src.utils.logs import get_logger


def create_features(config_path: Text):
    """Create features

    Args:
        config_path (Text): _description_
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("FEATURE_ENG", log_level=config["base"]["log_level"])

    # Load labeled data
    df = pd.read_csv(config["data_load"]["labeled_df"])

    # Create features and clean
    logger.info("Create ratios")
    df = preprocess.create_ratios(df)

    logger.info("Create delta days")
    df = preprocess.create_delta_days(df)

    logger.info("Create polynomial features")
    df = preprocess.create_poly_features(df, config)

    logger.info("Select features")
    df_selected = df[
        config["featurize"]["selected_features"]
        + ["date", "Data da coleta", "Resultado"]
    ]

    logger.info("Save data with selected features")
    df_selected.to_csv(config["featurize"]["ft_data_path"], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    create_features(config_path=args.config)
