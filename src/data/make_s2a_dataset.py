from typing import List

import boto3
import ee
import pandas as pd
import yaml

from src.data import in_out

ee.Initialize()


def create_region(lat, lon, buffer_size):
    amostragem_ponto = ee.Geometry.Point(lon, lat)
    return amostragem_ponto.buffer(buffer_size)


with open("params.yaml") as config_file:
    config = yaml.safe_load(config_file)

lat = config["data_create"]["latitude"]
lon = config["data_create"]["longitude"]
buffer_size = config["data_create"]["buffer_metros"]

region = create_region(lat, lon, buffer_size)


def reduce(img):
    """Extrai média, mediana, mínimo, máximo e desvio padrão de uma banda"""
    serie_reduce = img.reduceRegions(
        **{
            "collection": region,
            "reducer": ee.Reducer.mean()
            .combine(**{"reducer2": ee.Reducer.min(), "sharedInputs": True})
            .combine(**{"reducer2": ee.Reducer.max(), "sharedInputs": True})
            .combine(**{"reducer2": ee.Reducer.median(), "sharedInputs": True})
            .combine(**{"reducer2": ee.Reducer.stdDev(), "sharedInputs": True}),
            "scale": 20,
        }
    )

    serie_reduce = serie_reduce.map(lambda f: f.set({"millis": img.get("millis")})).map(
        lambda f: f.set({"date": img.get("date")})
    )

    return serie_reduce.copyProperties(img, ["system:time_start"])


def create_df(img, band):
    """Cria um dataframe para dados extraidos de imageCollection"""
    reduced_img = (
        img.select(band)
        .map(reduce)
        .flatten()
        .sort("date", True)
        .select(["millis", "date", "min", "max", "mean", "median", "stdDev"])
    )

    lista_df = (
        reduced_img.reduceColumns(
            ee.Reducer.toList(7),
            ["millis", "date", "min", "max", "mean", "median", "stdDev"],
        )
        .values()
        .get(0)
    )

    df = pd.DataFrame(
        lista_df.getInfo(),
        columns=["millis", "date"]
        + [band + "_" + stat for stat in ["min", "max", "mean", "median", "stdDev"]],
    )

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df


def add_id_date(img):
    """Adiciona ID e data da imagem Sentinel 2A"""

    return (
        img.set({"ID": img.get("system:id")})
        .set({"millis": img.date().millis()})
        .set("date", img.date().format())
    )


def mask_s2a_clouds(img):
    """Adiciona máscara de núvens em imagem S2A, retornando reflectância bandas B*

    Os bits 10 e 11 são nuvens e cirros, respectivamente.
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
    Créditos: Scripts Remote ( https://linktr.ee/scriptsremotesensing)"""

    qa60 = img.select("QA60")

    cloudBitMask = 1 << 10  # Bit 10 corresponde a nuvens
    cirrusBitMask = 1 << 11  # Bit 11 corresponde a cirrus

    mask = qa60.bitwiseAnd(cloudBitMask).eq(0) and (
        qa60.bitwiseAnd(cirrusBitMask).eq(0)
    )

    img_reflectancia = img.divide(10_000)

    return (
        img_reflectancia.updateMask(mask)
        .select("B.*")
        .copyProperties(img, img.propertyNames())
    )


def add_s2a_ndci_ndvi(img):
    """Adiciona NDCI e NDVI para imageCollection Sentinel 2A em uma região definida

    O cálculo dos índices considera as seguintes referências:
    - http://www.pjoes.com/pdf-98994-42186?filename=Assessing%20Spectral.pdf
    - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands
    - https://www.mdpi.com/2072-4292/13/15/2874/html
    """

    ndci_band = img.normalizedDifference(["B5", "B4"]).rename("NDCI")
    ndvi_band = img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    img_with_bands = (
        img.addBands([ndci_band, ndvi_band])
        .copyProperties(img, ["system:time_start"])
        .set("date", img.date().format("YYYY-MM-dd"))
    )

    return img_with_bands


def get_img_collection(config_path: str = "params.yaml"):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    lat = config["data_create"]["latitude"]
    lon = config["data_create"]["longitude"]
    buffer_size = config["data_create"]["buffer_metros"]

    region = create_region(lat, lon, buffer_size)

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2a_clouds)
        .map(lambda img: img.clip(region))
        .map(add_s2a_ndci_ndvi)
        .map(add_id_date)
    )


def get_df_from_img_collection(
    img_collection,
    config_path: str = "params.yaml",
    bands: List[str] = [
        "NDVI",
        "NDCI",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ],
):
    """Cria DataFrame com dados de NDVI, NDCI e bandas do S2A para ponto escolhido

    Refs:
    https://worldbank.github.io/OpenNightLights/tutorials/mod6_3_intro_to_sentinel2.html
    """

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    dfs = []
    for band in bands:
        formated = create_df(img_collection, band)
        dfs.append((band, formated))

    for count, df_tuple in enumerate(dfs):
        if count == 0:
            df = df_tuple[1]
        else:
            df = pd.merge(df, df_tuple[1], on=["millis", "date"])

    return df


if __name__ == "__main__":
    s3_path = config["data_load"]["s2a_df"]
    bucket, file_path = in_out.get_bucket_and_key_from_s3path(s3_path)
    s3_resource = boto3.resource("s3")

    s2a_collection = get_img_collection()
    df_gee = get_df_from_img_collection(s2a_collection)
    in_out.save_file_in_s3(df_gee, bucket, file_path, s3_resource)
