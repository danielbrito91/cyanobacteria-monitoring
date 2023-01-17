import datetime

import ee
import geemap
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.data import make_s2a_dataset


def cluster_gee(config: dict) -> pd.DataFrame:
    gee = pd.read_csv(config["data_load"]["s2a_df"])
    gee_id = gee["date"]
    gee_fts = gee.iloc[:, 2:]

    km = KMeans(
        n_clusters=config["gee_clean"]["kmeans"]["n_clusters"],
        random_state=config["base"]["random_state"],
    ).fit(gee_fts)
    gee_clusters = pd.DataFrame({"date": gee_id, "labels": km.labels_})

    return pd.concat([gee_clusters, gee_fts], axis=1)


def hard_code_cluster(clusters_fts, config) -> pd.DataFrame:

    clusters_fts.loc[
        (
            pd.to_datetime(clusters_fts["date"]).dt.date.isin(
                pd.to_datetime(config["gee_clean"]["imgs_com_nuvens"]).date
            )
        ),
        "labels",
    ] = 1

    return clusters_fts


def split_data_cloud(clusters_fts: pd.DataFrame, clean_cluster: list):

    gee_train = clusters_fts.loc[
        pd.to_datetime(clusters_fts["date"]) <= pd.to_datetime("2021-07-01")
    ]
    gee_test = clusters_fts.loc[
        pd.to_datetime(clusters_fts["date"]) > pd.to_datetime("2021-07-01")
    ]

    X_train = gee_train.iloc[:, 2:]
    X_test = gee_test.iloc[:, 2:]
    y_train = (gee_train["labels"].isin(clean_cluster)) * 1
    y_test = (gee_test["labels"].isin(clean_cluster)) * 1

    return X_train, X_test, y_train, y_test


def train_cloud_classifier(X_train, y_train):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    return clf


def flag_img_without_clouds(clf_path: str, s2a: pd.DataFrame) -> pd.DataFrame:
    """Predict if the image is without clouds

    Args:
        clf_path (str): where the cloud classifier is stored
        s2a (pd.DataFrame): S2A band values

    Returns:
        pd.DataFrame: s2a band values classified
    """
    clf = joblib.load(clf_path)

    s2a_fts = s2a.iloc[:, 2:]

    clean_img = clf.predict(s2a_fts)

    return pd.concat([s2a, pd.Series(clean_img, name="clean_img")], axis=1)


def plot_img(data_examinada, config, n_days=7):
    buffer = make_s2a_dataset.create_region(
        config["data_create"]["latitude"], config["data_create"]["longitude"], 150
    )

    data_futura = datetime.datetime.strptime(
        data_examinada, "%Y-%m-%d"
    ).date() + datetime.timedelta(days=n_days)
    data_futura = data_futura.strftime("%Y-%m-%d")

    s2a_img = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(buffer)
        .filterDate(data_examinada, data_futura)
        .map(make_s2a_dataset.mask_s2a_clouds)
        .map(make_s2a_dataset.add_s2a_ndci_ndvi)
        # .map(lambda img: img.clip(buffer))
    )

    Map = geemap.Map(
        location=[
            config["data_create"]["latitude"],
            config["data_create"]["longitude"],
        ],
        zoom_start=14,
    )
    Map.addLayer(
        s2a_img.first(), {"bands": ["B4", "B3", "B2"], "min": 0.01, "max": 0.2}, "RGB"
    )

    ponto = pd.DataFrame(
        {
            "latitude": [config["data_create"]["latitude"]],
            "longitude": [config["data_create"]["longitude"]],
        }
    )

    Map.addLayer(buffer, opacity=0.5)

    ms = s2a_img.first().date().millis().getInfo()

    print(datetime.datetime.fromtimestamp(ms / 1000.0))


    return Map
