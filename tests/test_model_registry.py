
import pandas as pd
from pydantic import BaseModel
import uvicorn, os
import mlflow, pickle
import json, joblib
from pathlib import Path
from mlflow import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, EnvironmentCredential
from dotenv import load_dotenv

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

def load_model_info(file_path):
    with open(file_path) as f:
        return json.load(f)
    
    
run_info = load_model_info("run_information.json")
model_name = run_info['model_name']

final_stage = 'production'

# Initialize the MLflow Client
client = MlflowClient()

latest_versions = client.search_model_versions(f"name='{model_name}'")
for i in latest_versions:
    print("versions ",i)
    print("\n")
latest_version  = latest_versions[0].version if latest_versions else None 

assert latest_version is not None , f"No model at {final_stage} stage"

# load model 
model_path = f"models:/{model_name}/{latest_version}"

# load latest model from model registry
model = mlflow.sklearn.load_model(model_path)

assert model is not None, "Failed to load model from registry"
print(f"The {model_name} with version {latest_version} was loaded successfully")

