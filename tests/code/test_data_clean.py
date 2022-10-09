import pandas as pd
import yaml


def test_cleaned_data():

    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    raw_gee = pd.read_csv(config["data_load"]["s2a_df"])
    cleaned_gee = pd.read_csv(config["gee_clean"]["clean_data_path"])

    assert raw_gee.shape[0] > cleaned_gee.shape[0]