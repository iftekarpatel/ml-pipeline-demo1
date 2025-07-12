import numpy as np
import pandas as pd 
import pickle

from sklearn.ensemble import RandomForestClassifier

#fetch the data from data/features directory
train_data = pd.read_csv('./data/features/train_features.csv')

X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values    

# Define and train the Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)   
model.fit(X_train, y_train) 

# Save the trained model to a file
pickle.dump(model, open('./models/random_forest_model.pkl', 'wb'))