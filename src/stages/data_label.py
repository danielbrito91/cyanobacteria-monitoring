import pandas as pd
import yaml
import argparse
from typing import Text

from src.utils.logs import get_logger
from src.features.featurize import clean_and_select_data
from src.data.label_gee import load_data, get_intervals

def data_label(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    logger = get_logger("DATA_LABEL", log_level=config["base"]["log_level"])

    gee, ciano_labels = load_data(config)
    gee["interval"]= get_intervals(gee["date"], min(gee["date"]), config)

    logger.info(f"Labeling the data ({config['data_create']['delta_dias']}-day interval)")
    ciano_labels["interval"] = get_intervals(ciano_labels["Data da coleta"], min(gee["date"]), config)
    df = pd.merge(gee, ciano_labels, on="interval")

    logger.info("Clean and select columns")
    selected_columns = (config["featurize"]["selected_clean_columns"] +
        [config["featurize"]["target_column"], 'Data da coleta', 'interval'])

    labeled_df = clean_and_select_data(df, selected_columns)

    logger.info("Save the labeled data")
    labeled_df.to_csv(config["data_load"]["labeled_df"], index=False)

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_label(config_path=args.config)