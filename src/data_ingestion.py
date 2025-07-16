import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Any, Dict, Tuple
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(params_path) as f:
            params = yaml.safe_load(f)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logging.error(f"{params_path} not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def fetch_data(url: str) -> pd.DataFrame:
    """Fetch data from a CSV URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data fetched from {url}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame: drop tweet_id, filter sentiments, encode labels."""
    try:
        df = df.drop(columns=['tweet_id'])
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Data preprocessing completed")
        return df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(
    df: pd.DataFrame, 
    test_size: float, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train and test sets."""
    try:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return train, test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    data_path: str = "data/raw"
) -> None:
    """Save train and test DataFrames to CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info(f"Train and test data saved to {data_path}")
    except Exception as e:
        logging.error(f"Error saving data to {data_path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params()
        test_size = params["data_ingestion"]["test_size"]
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = fetch_data(url)
        processed_df = preprocess_data(df)
        train_data, test_data = split_data(processed_df, test_size)
        save_data(train_data, test_data)
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()