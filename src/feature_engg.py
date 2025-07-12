import numpy as np
import pandas as pd 

import os

from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/processed directory
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# 🔧 Remove rows with NaN in 'content'
train_data = train_data.dropna(subset=['content'])
test_data = test_data.dropna(subset=['content'])

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df['label'] = y_test   

# store the processed data in the data/features directory
data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train_features.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_features.csv"), index=False)
