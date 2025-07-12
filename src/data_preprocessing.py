import numpy as np
import pandas as pd     

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/raw directory
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

# transform the text data

# Download required NLTK resources for lemmatization and stopwords
nltk.download('wordnet')
nltk.download('stopwords')

# Lemmatize each word in the input text
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()  # Split text into words
    text = [lemmatizer.lemmatize(y) for y in text]  # Lemmatize each word
    return " ".join(text)  # Join words back into a string

# Remove English stopwords from the input text
def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]  # Filter out stopwords
    return " ".join(Text)

# Remove all numeric characters from the input text
def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# Convert all words in the input text to lowercase
def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

# Remove punctuation and extra whitespace from the input text
def removing_punctuations(text):
    # Remove specified punctuation characters
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    # Remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

# Remove URLs from the input text
def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Replace sentences with less than 3 words in the DataFrame with NaN
def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

# Apply all normalization steps to the 'content' column of a DataFrame
def normalize_text(df):
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df

# Apply all normalization steps to a single sentence
def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return

# Apply normalization to the training and testing data
train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

# Save the normalized training and testing data to CSV files
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)       

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
