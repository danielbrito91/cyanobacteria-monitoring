import argparse
from typing import Text

import pandas as pd
import yaml

from src.data.load_vigilancia import load_vigilancia
from src.data.make_s2a_dataset import gee_to_df
from src.utils.logs import get_logger


def data_load(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("DATA_LOAD", log_level=config["base"]["log_level"])

    logger.info("Reading data from SISAGUA (Datagov)")
    vigi = load_vigilancia(config_path)

    logger.info("Save data from SISAGUA (Datagov)")
    vigi.to_csv(config["data_load"]["labels_df"], index=False)

    logger.info("Reading data from GEE")
    gee = gee_to_df(config_path)

    logger.info("Save data from GEE")
    gee.to_csv(config["data_load"]["s2a_df"], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
