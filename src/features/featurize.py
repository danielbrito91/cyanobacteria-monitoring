import pandas as pd
from typing import Text

def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["B3_B2"] = df["B3_median"] / df["B2_median"]
    df["B5_B2"] = df["B5_median"] / df["B2_median"]
    df["B3_B4"] = df["B3_median"] / df["B4_median"]
    df["B5_B4"] = df["B5_median"] / df["B4_median"]
    df["B5B3_B2"] = (df["B5_median"] + df["B3_median"]) / df["B2_median"]

    return df

def clean_and_select_data(df, selected_columns):

    df_filtered = df.loc[
        (df["NDVI_median"]<-.05) &
        (df["B1_median"]< 0.5) &
        (df["B2_median"]< 0.6)]
    
    df_final = df_filtered[selected_columns].reset_index(drop = True)

    return df_final