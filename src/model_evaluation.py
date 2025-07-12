import numpy as np
import pandas as pd 
import os
import pickle   
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Load the trained model
model = pickle.load(open('./models/random_forest_model.pkl', 'rb')) 
# Fetch the test data from the features directory
test_data = pd.read_csv('./data/features/test_features.csv')
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC AUC:", roc_auc)      

# Save the evaluation results to a CSV file
metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc
}
with open('metrics_dict.json', 'w') as f:
    json.dump(metrics_dict, f)

