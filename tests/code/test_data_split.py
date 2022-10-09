import pandas as pd
import yaml


def test_data_leak_in_test_data():

    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)

    train = pd.read_csv(config["data_split"]["trainset_path"])
    test =  pd.read_csv(config["data_split"]["testset_path"])

    concat = pd.concat([train, test])
    concat.drop_duplicates(inplace=True)

    assert concat.shape[0] == train.shape[0] + test.shape[0]
    assert max(train["date"]) < min(test["date"])