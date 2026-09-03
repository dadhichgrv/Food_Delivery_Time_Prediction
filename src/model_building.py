import os
import mlflow, sklearn
import mlflow.sklearn
import pandas as pd
import pickle
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from azure.identity import DefaultAzureCredential , InteractiveBrowserCredential
from azure.ai.ml import MLClient

load_dotenv()

# authenticate
tenant_id = "2c06054d-006e-4d5f-bf87-12294be4e2da"

try:
    credential = DefaultAzureCredential(interactive_browser_tenant_id=tenant_id)
    credential.get_token("https://management.azure.com/.default")
except Exception:
    credential = InteractiveBrowserCredential(tenant_id=tenant_id)


ml_client = MLClient(
    credential = credential,
    subscription_id = os.getenv("SUBSCRIPTION_ID"),
    resource_group_name = os.getenv("RESOURCE_GROUP"),
    workspace_name = os.getenv("ML_WORKSPACE_NAME")
)

mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
print(mlflow_tracking_uri)
mlflow.set_tracking_uri(mlflow_tracking_uri)


# Read data
train = pd.read_csv("./data/features/train_features.csv")

# split into X and y
X_train_trans = train.drop(columns='time_taken')
y_train_pt = train['time_taken']

mlflow.set_experiment('linear-regression-baseline-autolog')

with mlflow.start_run() as run:

    # Train the model
    lr = LinearRegression()
    lr.fit(X_train_trans,y_train_pt)

    # Save the model
    pickle.dump(lr, open('model.pkl','wb'))

    mlflow.sklearn.log_model(lr,"linear_regression")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

