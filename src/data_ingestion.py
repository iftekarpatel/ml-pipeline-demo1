import numpy as np
import pandas as pd

import os

from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
# delete tweet id
df.drop(columns=['tweet_id'],inplace=True)

# Filter the DataFrame to keep only rows where sentiment is 'happiness' or 'sadness'
final_df = df[df['sentiment'].isin(['happiness','sadness'])]

# Replace 'happiness' with 1 and 'sadness' with 0 in the sentiment column (binary encoding)
final_df['sentiment'].replace({'happiness':1, 'sadness':0}, inplace=True)

# Split the filtered and encoded DataFrame into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

# Save the training and testing data to CSV files
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
