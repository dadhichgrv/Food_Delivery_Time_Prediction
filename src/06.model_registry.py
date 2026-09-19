import os
import mlflow
import pandas as pd
import pickle
import json
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient
from pathlib import Path
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, EnvironmentCredential
from azure.ai.ml import MLClient
import logging, joblib

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

# logging configure
logger = logging.getLogger('model_registry')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# authenticate
tenant_id = os.getenv("tenant_id")

def load_model_info(file_path):
    with open(file_path) as f:
        return json.load(f)
    
if __name__=="__main__":
    ROOT_FOLDER = Path(__file__).resolve().parent.parent 
    model_path = ROOT_FOLDER / "models" / "random_forest.joblib"
    run_info_path = ROOT_FOLDER / "run_information.json"

    # register the model 
    run_info = load_model_info(run_info_path)

    run_id = run_info['run_id']
    model_name = run_info['model_name']

    model_registry_path = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(model_uri=model_registry_path,
                                          name = model_name)
    
    # get registered model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name 
    logger.info(f"The latest model version is {registered_model_version}")

    # update the model stage to staging
    client = MlflowClient()
    client.set_model_version_tag(
        name=registered_model_name,
        version=str(registered_model_version),
        key="stage",
        value="production"
    )
    
    logger.info(f"Model version {registered_model_version} aliased to 'production'")

  