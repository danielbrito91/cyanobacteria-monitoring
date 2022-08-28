import argparse
import pandas as pd
import yaml
from typing import Text
from dateutil.relativedelta import relativedelta
import argparse
import smogn

from src.utils.logs import get_logger

def oversampling(df_train, config) -> pd.DataFrame:
    
    return smogn.smoter(data = df_train,
        y = config["featurize"]["target_column"],
        k = 5,
        samp_method="extreme",
        rel_thres=.9,
        rel_method="auto",
        rel_xtrm_type="high",
        rel_coef=10)

def data_split(config_path: Text) -> None:
    
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    logger = get_logger("DATA_SPLIT", log_level=config["base"]["log_level"])

    logger.info("Load features")
    df = pd.read_csv(config["data_load"]["labeled_df"])
    
    date_limit_test = (pd.to_datetime(max(df["date"])) - relativedelta(years=config["data_split"]["years_split_test"]))

    logger.info("Split data into train and test")
    df_train = df[pd.to_datetime(df["date"]) <= date_limit_test]
    df_test = df.loc[pd.to_datetime(df["date"]) > date_limit_test]

    logger.info("Oversampling training data")
    df_train = oversampling(df_train, config)

    logger.info("Save train and test sets")
    df_train.to_csv(config["data_split"]["trainset_path"], index=False)
    df_test.to_csv(config["data_split"]["testset_path"], index=False)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config) 