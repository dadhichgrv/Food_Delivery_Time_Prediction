import os
import mlflow, sklearn
import mlflow.sklearn
import pandas as pd
import pickle
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yaml
from azure.identity import DefaultAzureCredential , InteractiveBrowserCredential
from azure.ai.ml import MLClient
from sklearn.model_selection import RandomizedSearchCV

load_dotenv()

# authenticate
tenant_id = os.getenv("tenant_id")

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

params_path = "params.yaml"
with open(params_path, 'r') as file:
    params = yaml.safe_load(file)  

# Access the required value
max_depth = params['model_building']['max_depth']
n_estimators = params['model_building']['n_estimators']

# Read data
train = pd.read_csv("./data/features/train_features.csv")

# split into X and y
X_train_trans = train.drop(columns='time_taken')
y_train_pt = train['time_taken']

mlflow.set_experiment('random_forest')


# Train the model
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'max_depth':[3,4,5],
    'n_estimators':[100,200,300]
            }

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,cv=5, random_state=42)

with mlflow.start_run() as run:

    random_search.fit(X_train_trans,y_train_pt)

    best_params = random_search.best_params_
    best_score  = random_search.best_score_
    scorer     = random_search.scorer_

    # Save the model
    pickle.dump(random_search.best_estimator_, open('best_rf_model.pkl','wb'))

    mlflow.sklearn.log_model(random_search.best_estimator_,"random_forest")
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score",best_score)
    
    # Set tags
    mlflow.set_tag('author','GD')
    mlflow.set_tag('model','Random Forest')

    # logging dataset
    train_mlflow = mlflow.data.from_pandas(train, name="train_dataset")
    mlflow.log_input(train_mlflow,context="training dataset")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

