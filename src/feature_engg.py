import os
import yaml
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Train data loaded from {train_path}")
        logging.info(f"Test data loaded from {test_path}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN in 'content'."""
    before = len(df)
    df = df.dropna(subset=['content'])
    after = len(df)
    logging.info(f"Removed {before - after} rows with NaN content.")
    return df

def extract_features(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    max_features: int
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Apply Bag of Words (CountVectorizer) to train and test data."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Feature extraction with CountVectorizer completed.")
        return X_train_bow.toarray(), X_test_bow.toarray(), vectorizer
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        raise

def save_features(
    X: np.ndarray, 
    y: np.ndarray, 
    file_path: str
) -> None:
    """Save features and labels to a CSV file."""
    try:
        df = pd.DataFrame(X)
        df['label'] = y
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Features saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving features to {file_path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        max_features = params["feature_engg"]["max_features"]

        train_data, test_data = load_data(
            './data/processed/train_processed.csv',
            './data/processed/test_processed.csv'
        )

        train_data = clean_data(train_data)
        test_data = clean_data(test_data)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow, X_test_bow, _ = extract_features(X_train, X_test, max_features)

        save_features(X_train_bow, y_train, os.path.join("data", "features", "train_features.csv"))
        save_features(X_test_bow, y_test, os.path.join("data", "features", "test_features.csv"))

        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()