import os
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score

# Load original and transformed features for test data
test = pd.read_csv('./data/features/test.csv')
test_features = pd.read_csv('./data/features/test_features.csv')

X_test_trans = test_features.drop(columns='time_taken',axis=1)

y_test = test['time_taken']

pt = pickle.load(open("power_transformer.pkl","rb"))
lr = pickle.load(open("model.pkl","rb"))

y_pred_test = lr.predict(X_test_trans)
y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

mae = mean_absolute_error(y_test,y_pred_test_org)
r2score = r2_score(y_test,y_pred_test_org)

metrics = {"mean_square_error":mae,
           "r2_score":r2score}

json.dump(metrics,open("metrics.json","w"))

