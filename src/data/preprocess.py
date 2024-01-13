from typing import Union

import pandas as pd
import smogn
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import PolynomialFeatures


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

    n_train = df_train.shape[0]
    n_test = df_test.shape[0]

    print(f"{n_test/ (n_train + n_test):.2%} dos dados utilizados para teste")

    return df_train, df_test


def create_delta_days(df: pd.DataFrame) -> pd.DataFrame:
    # Add a column with day difference between S2A and SISAGUA
    df["delta_days"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["Data da coleta"])
    ).dt.days

    return df


def create_poly_features(df: pd.DataFrame, config: dict, labeled=True) -> pd.DataFrame:
    poly = PolynomialFeatures(
        degree=config["featurize"]["poly_degree"], include_bias=False
    )
    if labeled:
        target_ids = df[["date", "Resultado", "Data da coleta", "interval"]]
        X = df.drop(columns=["date", "Resultado", "Data da coleta", "interval"])
    else:
        target_ids = df[["date"]]
        X = df.drop(columns="date")

    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names(X.columns))

    return pd.concat([target_ids, X_poly], axis=1)
