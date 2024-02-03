import sys
from datetime import date
from typing import Text, List

import boto3
import joblib
import numpy as np
import pandas as pd
import logging

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from io import StringIO, BytesIO
from sklearn.preprocessing import PolynomialFeatures

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])


sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
job.commit()

logger = glueContext.get_logger()

# CTEs
bucket = "cyano-guaiba"
s3_client = boto3.client("s3")
model_path_prefix = "model/full_model.joblib"
s3_resource = boto3.resource("s3")
selected_cols = [
    "B1_median B5B3_B2" ,
    "B3_B2 B5B3_B2" ,
    "B3_B4" ,
    "B3_B4 B5B3_B2" ,
    "B4_median B3_B2" ,
    "B4_median B5B3_B2" ,
    "B5B3_B2" ,
    "B5B3_B2^2" ,
    "B5_B4^2" ,
    "B5_median B5B3_B2" ,
    "B6_median^2" ,
    "B8A_median B5B3_B2" ,
    "B9_median B3_B4" ,
    "B9_median B5B3_B2" ,
    "B11_median^2",
    "delta_days",
    "NDCI_median B5_B4" ,
    "NDVI_median B5_B2"
]
gee_path = "s3://gee-guaiba/latitude=-30.012175, longitude=-51.215679, buffer=20/gee.gzip"
    
def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["B3_B2"] = df["B3_median"] / df["B2_median"]
    df["B5_B2"] = df["B5_median"] / df["B2_median"]
    df["B3_B4"] = df["B3_median"] / df["B4_median"]
    df["B5_B4"] = df["B5_median"] / df["B4_median"]
    df["B5B3_B2"] = (df["B5_median"] + df["B3_median"]) / df["B2_median"]

    return df

def create_poly_features(df: pd.DataFrame, degree: int = 2, labeled=True) -> pd.DataFrame:
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
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

def get_bucket_and_key_from_s3path(s3_path: str):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    return bucket, key

def save_file_in_s3(df: pd.DataFrame, bucket: str, file_path: str, s3_resource) -> None:
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression="gzip")
    s3_resource.Object(bucket, file_path).put(Body=parquet_buffer.getvalue())

def get_model(s3_client, bucket, model_path_prefix):
    response = s3_client.get_object(
        Bucket=bucket,
        Key=model_path_prefix)
        
    joblib_content = response["Body"].read()
    return joblib.load(BytesIO(joblib_content))

def create_fts(gee_path):
    gee = pd.read_parquet(gee_path)
    gee_fts = (
        gee.pipe(create_ratios)
        .assign(delta_days=0)
        .pipe(create_poly_features, labeled=False)
    )

    return gee_fts[selected_cols], gee_fts["date"]

def predict(model, X, id) -> pd.DataFrame:
    y_pred = model.predict(X)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    return pd.DataFrame({"date": id, "y_pred": y_pred})

def persist(df_predicted) -> None:
    today_ = date.today().strftime("%Y%m%d")
    s3_path = f"s3://cyano-guaiba/data/prediction_{today_}.gzip"
    df_predicted.to_parquet(s3_path)

if __name__ == "__main__":
    
    logging.info("Get model")
    model = get_model(s3_client, bucket, model_path_prefix)

    logging.info("Create features")
    X, id = create_fts(gee_path)

    logging.info("Predict")
    df_predicted = predict(model, X, id)
    
    logging.info("Persist")
    persist(df_predicted) 