import smogn
import pandas as pd
from typing import Union
from dateutil.relativedelta import relativedelta

# poly
# transformer
# feature selection


def oversampling(df_train, config):

    return smogn.smoter(
        data=df_train,
        y=config["featurize"]["target_column"],
    )


def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["B3_B2"] = df["B3_median"] / df["B2_median"]
    df["B5_B2"] = df["B5_median"] / df["B2_median"]
    df["B3_B4"] = df["B3_median"] / df["B4_median"]
    df["B5_B4"] = df["B5_median"] / df["B4_median"]
    df["B5B3_B2"] = (df["B5_median"] + df["B3_median"]) / df["B2_median"]

    return df


def clean_data(df):

    return df.loc[
        (df["NDVI_median"] < -0.05) & (df["B1_median"] < 0.5) & (df["B2_median"] < 0.6)
    ]


def train_test_split_ts(
    df: pd.DataFrame, config: dict
) -> Union[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        df (pd.DataFrame): cleaned and featurized dataframe
        config (dict): _description_

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: _description_
    """

    date_limit_test = pd.to_datetime(max(df["date"])) - relativedelta(
        years=config["data_split"]["years_split_test"]
    )

    df_train = df[pd.to_datetime(df["date"]) <= date_limit_test]
    df_test = df.loc[pd.to_datetime(df["date"]) > date_limit_test]

    return df_train, df_test


def create_delta_days(df):
    # Add a column with day difference between S2A and SISAGUA
    pass
