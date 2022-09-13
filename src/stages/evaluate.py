import pandas as pd
import yaml
from typing import Text
import joblib
import argparse
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import json
from pathlib import Path
import mlflow
import mlflow.xgboost
import mlflow.sklearn

from src.utils.logs import get_logger
from src.features.featurize import create_ratios
from src.report.plot_predicted_vs_labels import plot_predicted_vs_labels

def load_and_featurize(config: Text):

    df_test = pd.read_csv(config["data_split"]["testset_path"])
    selected_cols = config["featurize"]["selected_features"]
    target = config["featurize"]["target_column"]
    df_test = create_ratios(df_test)
    
    id_test = df_test[["date", "Data da coleta"]]

    df_test = df_test[selected_cols + [target]]
    X_test = df_test.drop(columns = [target])
    y_test = df_test[target]

    return id_test, X_test, y_test

def evaluate_model(config_path: Text) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    logger = get_logger("EVALUATE", log_level=config["base"]["log_level"])

    logger.info("Load model")
    model_path = config["train"]["model_path"]
    model = joblib.load(model_path)

    logger.info("Load test dataset")
    id_test, X_test, y_test = load_and_featurize(config)

    logger.info("Predict")
    y_pred = model.predict(X_test)

    logger.info("Evaluating")
    assert len(y_test.values) == len(y_pred)

    df_plot = pd.DataFrame({
        "date": id_test["date"],
        "amostragem": id_test["Data da coleta"],
        "y_true": y_test,
        "y_pred": y_pred
    })

    logger.info(f"Save plot to {config['evaluate']['predicted_vs_fitted_plot']}")
    plot_predicted_vs_labels(df_plot).write_image(config['evaluate']['predicted_vs_fitted_plot'])  

    logger.info("Save metrics")
    report = {
        "mae": mean_absolute_error(y_test, y_pred)
    }

    mlflow.set_experiment(experiment_name=config["mlflow_config"]["experiment_name"])
    with mlflow.start_run(run_name=config["mlflow_config"]["run_name"]):
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_artifacts(model)


    json.dump(
        obj={"mae": report["mae"]},
        fp=open(config["evaluate"]["metrics_file"], "w")
    )
   
    logger.info("Save predicted test data")
    df_plot.to_csv(config["evaluate"]["metrics_data"], index=False)

if __name__ == "__main__":
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)