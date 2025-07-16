import os
import re
import logging
from typing import Any
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def download_nltk_resources() -> None:
    """Download required NLTK resources."""
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def lemmatization(text: str) -> str:
    """Lemmatize each word in the input text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text: str) -> str:
    """Remove English stopwords from the input text."""
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in str(text).split() if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text: str) -> str:
    """Remove all numeric characters from the input text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Convert all words in the input text to lowercase."""
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    """Remove punctuation and extra whitespace from the input text."""
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    return " ".join(text.split()).strip()

def removing_urls(text: str) -> str:
    """Remove URLs from the input text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame, min_words: int = 3) -> pd.DataFrame:
    """Replace sentences with less than min_words in the DataFrame with NaN."""
    try:
        mask = df['content'].apply(lambda x: len(str(x).split()) < min_words)
        df.loc[mask, 'content'] = np.nan
        logging.info(f"Small sentences (less than {min_words} words) replaced with NaN.")
        return df
    except Exception as e:
        logging.error(f"Error removing small sentences: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization steps to the 'content' column of a DataFrame."""
    try:
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def main() -> None:
    try:
        download_nltk_resources()
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        train_data = remove_small_sentences(train_data)
        test_data = remove_small_sentences(test_data)

        save_data(train_data, os.path.join("data", "processed", "train_processed.csv"))
        save_data(test_data, os.path.join("data", "processed", "test_processed.csv"))
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    main()