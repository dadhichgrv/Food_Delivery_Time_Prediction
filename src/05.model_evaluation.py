import os
import mlflow
import pandas as pd
import pickle
import json
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
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
mlflow.set_experiment('DVC Pipeline')

# logging configure
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# authenticate
tenant_id = os.getenv("tenant_id")

# load the run id from training
# with open("run_id.txt") as f:
#     run_id = f.read().strip()

# Read data
def read_data(url:Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        logger.error("File Not found")
    return df


def load_model(model_path:Path):
    model = joblib.load(model_path)
    return model

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict={
        "run_id":run_id,
        "artifact_path":artifact_path,
        "model_name":model_name
    }

    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)


if __name__=="__main__":

    ROOT_FOLDER = Path(__file__).parent.parent
    INPUT_FOLDER = ROOT_FOLDER 
    test_data_path = ROOT_FOLDER / "data" / "features" / "test_features.csv"
    train_data_path = ROOT_FOLDER / "data" / "features" / "train_features.csv"
    model_path = ROOT_FOLDER / "models" / "random_forest.joblib"

    # Load original and transformed features for test data
    test = pd.read_csv(ROOT_FOLDER / "data" / "features" / "test.csv")
    test_features = pd.read_csv(test_data_path)

    train = pd.read_csv(ROOT_FOLDER / "data" / "features" / "train.csv")
    train_features = pd.read_csv(train_data_path)
    
    X_train_trans = train_features.drop(columns='time_taken')
    y_train = train['time_taken']
    X_test_trans = test_features.drop(columns='time_taken')
    y_test = test['time_taken']

    pt = pickle.load(open("power_transformer.pkl","rb"))
    model = load_model(model_path)
    logger.info("Model Loaded successfully")

    y_pred_test = model.predict(X_test_trans)
    y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

    y_pred_train = model.predict(X_train_trans)
    y_pred_train_org = pt.inverse_transform(y_pred_train.reshape(-1,1))
    
    train_mae = mean_absolute_error(y_train,y_pred_train_org)
    test_mae  = mean_absolute_error(y_test,y_pred_test_org)
    logger.info("Error Calculated")

    test_r2score = r2_score(y_test,y_pred_test_org)
    train_r2score  = r2_score(y_train,y_pred_train_org)

    cv_scores = cross_val_score(model, X_train_trans, y_train, cv=5, scoring = "neg_mean_absolute_error")
    logger.info("Cross validation Done")

    mean_cv_score = -(cv_scores.mean())


    with mlflow.start_run() as run:

        mlflow.set_tag("model","Food Delivery Time Regressor")
        mlflow.log_params(model.get_params())

        mlflow.log_metric("train_mae",train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2score",train_r2score)
        mlflow.log_metric("test_r2score",test_r2score)

        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})

        # logging dataset
        try:
            train_data_input = mlflow.data.from_pandas(train,targets="time_taken")
            test_data_input = mlflow.data.from_pandas(test,targets="time_taken")

            mlflow.log_input(dataset=train_data_input, context="training")
            mlflow.log_input(dataset=test_data_input, context="validation")

        except mlflow.exceptions.MlflowException as e:
            print(f"Warning: could not log dataset input (likely already registered): {e}")

        #mlflow.log_artifact(ROOT_FOLDER / "models" / "random_forest.joblib")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None # Keep None since you register it in script 06
        )
        mlflow.log_artifact(ROOT_FOLDER / "models" / "preprocessor.joblib")
        mlflow.log_artifact(ROOT_FOLDER / "power_transformer.pkl")

        artifact_uri = mlflow.get_artifact_uri()

        logger.info('MLFlow logging completed')

    run_id = run.info.run_id 
    model_name = "food_delivery_time_prediction_model"

    save_json_path = ROOT_FOLDER / "run_information.json"

    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")    


        
 

 

    # log into the SAME run as training
    # with mlflow.start_run(run_id=run_id):
    #     mlflow.log_metrics(metrics)

    #     # logging dataset
    #     test_mlflow = mlflow.data.from_pandas(test,name="test_dataset")
    #     mlflow.log_input(test_mlflow,context="test dataset")

    # RF : {"mean_square_error": 3.1250103792764397, "r2_score": 0.8250822570118966}
    # Test file is 7614 rows

