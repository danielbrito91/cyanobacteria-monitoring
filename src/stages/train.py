import argparse
from typing import Text
import tempfile
from pathlib import Path

import json
import joblib
import mlflow
import pandas as pd
import yaml

from src.data import preprocess
from src.train import train_model
from src.utils.logs import get_logger


def train(config_path: Text):
    """Create MLFlow Experiment and save the trained model

    Args:
        config_path (Text): _description_
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    # Load labeled data
    df = pd.read_csv(config["data_load"]["labeled_df"])

    # Create features and clean
    df = preprocess.clean_data(df)
    df = preprocess.create_ratios(df)

    # Train
    mlflow.set_experiment(experiment_name=config["mlflow_config"]["experiment_name"])
    with mlflow.start_run(run_name=config["train"]["estimator_name"]):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train_model.train_model(df, config)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        mlflow.log_metrics({"mae": performance["mae"]})
        mlflow.log_metrics({"mlse": performance["mlse"]})
        mlflow.log_params(artifacts["params"])

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["tscv"], Path(dp, "tscv.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            train_model.save_dict(artifacts["performance"], Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    logger.info("Save model")
    joblib.dump(artifacts["model"], config["train"]["model_path"])


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
