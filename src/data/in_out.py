from io import BytesIO

import pandas as pd


def get_bucket_and_key_from_s3path(s3_path: str):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    return bucket, key


def save_file_in_s3(df: pd.DataFrame, bucket: str, file_path: str, s3_resource) -> None:
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression="gzip")
    s3_resource.Object(bucket, file_path).put(Body=parquet_buffer.getvalue())


def read_file_from_s3_using_boto(bucket: str, path: str, s3_resource) -> pd.DataFrame:
    buffer = BytesIO()
    obj = s3_resource.Object(bucket, path)
    obj.download_fileobj(buffer)
    return pd.read_parquet(obj)
