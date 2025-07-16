import os
import pickle
import json
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path: str) -> Any:
    """Load a trained model from a file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def load_test_data(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data and split into features and labels."""
    try:
        test_data = pd.read_csv(test_path)
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        logging.info(f"Test data loaded from {test_path}")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error loading test data from {test_path}: {e}")
        raise

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        logging.info("Model evaluation completed.")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {file_path}: {e}")
        raise

def main() -> None:
    try:
        model_path = './models/random_forest_model.pkl'
        test_path = './data/features/test_features.csv'
        metrics_path = 'metrics_dict.json'

        model = load_model(model_path)
        X_test, y_test = load_test_data(test_path)
        metrics = evaluate_model(model, X_test, y_test)

        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        save_metrics(metrics, metrics_path)
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()