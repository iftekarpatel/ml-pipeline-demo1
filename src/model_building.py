import os
import yaml
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path) as f:
            params = yaml.safe_load(f)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def load_train_data(file_path: str) -> pd.DataFrame:
    """Load training data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Training data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def prepare_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split DataFrame into features and labels."""
    try:
        X = df.drop(columns=['label']).values
        y = df['label'].values
        logging.info("Features and labels prepared.")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing features and labels: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a Random Forest model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: Any, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        n_estimators = params["model_building"]["n_estimators"]
        max_depth = params["model_building"]["max_depth"]

        train_data = load_train_data('./data/features/train_features.csv')
        X_train, y_train = prepare_features_labels(train_data)
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, './models/random_forest_model.pkl')
        logging.info("Model building pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model building pipeline failed: {e}")

if __name__ == "__main__":
    main()