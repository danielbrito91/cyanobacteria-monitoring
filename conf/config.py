
from pathlib import Path
import mlflow

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "conf")
DATA_DIR = Path(BASE_DIR, "data")
LOGS_DIR = Path(BASE_DIR, "logs")

# Stores
MODEL_REGISTRY = Path("models")

# MLFlow models
#mlflow.set_tracking_uri("file:///" + str(MODEL_REGISTRY.absolute()))
mlflow.set_tracking_uri("sqlite:///:memory:")
