import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Read data
train = pd.read_csv("./data/features/train_features.csv")

# split into X and y
X_train_trans = train.drop(columns='time_taken',axis=1)
y_train_pt = train['time_taken']

# Train the model
lr = LinearRegression()
lr.fit(X_train_trans,y_train_pt)

# Save the model
pickle.dump(lr, open('model.pkl','wb'))

