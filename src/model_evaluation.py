import os
import mlflow
import pandas as pd
import pickle
import json
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, EnvironmentCredential
from azure.ai.ml import MLClient

load_dotenv()

tenant_id = os.getenv("tenant_id")

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception:
    credential = InteractiveBrowserCredential(tenant_id=tenant_id)

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("ML_WORKSPACE_NAME")
)

mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)

# load the run id from training
with open("run_id.txt") as f:
    run_id = f.read().strip()

# Load original and transformed features for test data
test = pd.read_csv('./data/features/test.csv')
test_features = pd.read_csv('./data/features/test_features.csv')

X_test_trans = test_features.drop(columns='time_taken')

y_test = test['time_taken']

pt = pickle.load(open("power_transformer.pkl","rb"))
model = pickle.load(open("best_rf_model.pkl","rb"))

y_pred_test = model.predict(X_test_trans)
y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

mae = mean_absolute_error(y_test,y_pred_test_org)
r2score = r2_score(y_test,y_pred_test_org)


metrics = {"mean_square_error":mae,
           "r2_score":r2score
          }

json.dump(metrics,open("metrics.json","w"))

# log into the SAME run as training
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(metrics)

    # logging dataset
    test_mlflow = mlflow.data.from_pandas(test,name="test_dataset")
    mlflow.log_input(test_mlflow,context="test dataset")

# RF : {"mean_square_error": 3.1250103792764397, "r2_score": 0.8250822570118966}
# Test file is 7614 rows

