import argparse
from typing import Text

import pandas as pd
import yaml

from src.data import preprocess
from src.utils import logs


def prep_and_split(config_path: Text):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = logs.get_logger("DATA_SPLIT", log_level=config["base"]["log_level"])

    logger.info("Load labeled data")
    df = pd.read_csv(config["data_load"]["labeled_df"])

    logger.info("Create features and clean the dataset")
    df = preprocess.clean_data(df)
    df = preprocess.create_ratios(df)

    logger.info("Split the dataset")
    df_train, df_test = preprocess.train_test_split_ts(df, config)

    logger.info("Persist the splitted dataset")
    df_train.to_csv(config["data_split"]["trainset_path"], index=False)
    df_test.to_csv(config["data_split"]["testset_path"], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    prep_and_split(config_path=args.config)
