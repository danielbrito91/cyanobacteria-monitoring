import argparse
import json
from typing import Text

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.data import clean_gee
from src.utils import logs


def train_gee_cleaner(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = logs.get_logger("TRAIN_CLEAN_GEE", log_level=config["base"]["log_level"])

    if config["gee_clean"]["train_new_clf"]:

        logger.info("Training a cloud classifier")
        logger.info("Clustering the image bands")
        clst = clean_gee.cluster_gee(config)

        logger.info("Hard code labels of images with clouds")
        clst_mod = clean_gee.hard_code_cluster(clst, config)

        clean_clst = config["gee_clean"]["kmeans"]["clean_cluster"]
        logger.info(f"Train data on cluster {clean_clst} (image without clouds)")
        X_train, X_test, y_train, y_test = clean_gee.split_data_cloud(
            clst_mod, clean_clst
        )
        cloud_clf = clean_gee.train_cloud_classifier(X_train, y_train)

        logger.info("Save model")
        clf_path = config["gee_clean"]["classifier"]["clf_path"]
        joblib.dump(cloud_clf, clf_path)
        y_pred = cloud_clf.predict(X_test)

        logger.info("Train data on cluster")
        clf_performance = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        perf_path = config["gee_clean"]["classifier"]["clf_performance"]
        with open(perf_path, "w") as write_file:
            json.dump(clf_performance, write_file)
    else:
        logger.info("Using model that was trained before")


def gee_clean_save(config_path: Text):

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = logs.get_logger("CLEAN_GEE", log_level=config["base"]["log_level"])

    logger.info("Classification")
    gee = pd.read_csv(config["data_load"]["s2a_df"])
    clf_path = config["gee_clean"]["classifier"]["clf_path"]
    gee_cloud = clean_gee.flag_img_without_clouds(clf_path, gee)
    gee_cloud_free = (
        gee_cloud.loc[gee_cloud["clean_img"] == 1]
        .drop(columns="clean_img")
        .reset_index(drop=True)
    )

    logger.info("Save classified data")
    gee_cloud_free.to_csv(config["gee_clean"]["clean_data_path"], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_gee_cleaner(config_path=args.config)
    gee_clean_save(config_path=args.config)
