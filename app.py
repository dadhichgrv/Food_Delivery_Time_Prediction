from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import uvicorn, os
import mlflow, pickle
import json, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from scripts.data_clean_utils import perform_data_cleaning
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


class Data(BaseModel):
    ID:str
    Delivery_person_ID:str
    Delivery_person_Age:int
    Delivery_person_Ratings:str
    Restaurant_latitude:float
    Restaurant_longitude:float
    Delivery_location_latitude:float
    Delivery_location_longitude:float
    Order_Date:str
    Time_Orderd:str
    Time_Order_picked:str
    Weatherconditions:str
    Road_traffic_density:str
    Vehicle_condition:int
    Type_of_order:str
    Type_of_vehicle:str
    multiple_deliveries:str
    Festival:str
    City:str


preprocessor_path = "models/preprocessor.joblib"



def load_model_info(file_path):
    with open(file_path) as f:
        return json.load(f)

def load_preprocessor(preprocessor_path:str):
    preprocessor= joblib.load(preprocessor_path)
    return preprocessor


num_cols = ["age","ratings","pick_up_time_mins","distance"]

nominal_cat_cols = ['weather','type_of_order',
                    'type_of_vehicle',"festival",
                    "city_type",
                    "is_weekend",
                    "time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

    
run_info = load_model_info("run_information.json")
model_name = run_info['model_name']

final_stage = 'production'

# Initialize the MLflow Client
client = MlflowClient()

# 1. Search for all versions of this model
model_versions = client.search_model_versions(f"name='{model_name}'")

# 2. Find the latest version that has the tag {'stage': 'production'}
latest_version = None
for mv in model_versions:
    # Check if the 'stage' key exists in tags and matches our target
    if mv.tags.get("stage") == final_stage:
        latest_version = mv.version
        break  # search_model_versions returns latest versions first

if latest_version is None:
    raise RuntimeError(f"No model version found with tag stage='{final_stage}'")

# load latest model from model registry
model_path = f"models:/{model_name}/{latest_version}"
model = mlflow.sklearn.load_model(model_path)


preprocessor = load_preprocessor(preprocessor_path)
pt = pickle.load(open("power_transformer.pkl","rb"))

model_pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ("regressor",model)
            ])


app = FastAPI()


@app.get("/")
def home():
    return "Welcome to food delivery time prediction application"


@app.post(path="/predict")
def post_prediction(X:Data):
    pred_data = pd.DataFrame({
    'id':X.ID,
    'delivery_person_id':X.Delivery_person_ID,
    'delivery_person_age':X.Delivery_person_Age,
    'delivery_person_ratings':X.Delivery_person_Ratings,
    'restaurant_latitude':X.Restaurant_latitude,
    'restaurant_longitude':X.Restaurant_longitude,
    'delivery_latitude':X.Delivery_location_latitude,
    'delivery_longitude':X.Delivery_location_longitude,
    'order_date':X.Order_Date,
    'order_time':X.Time_Orderd,
    'order_picked_time':X.Time_Order_picked,
    'weather':X.Weatherconditions,
    'traffic':X.Road_traffic_density,
    'vehicle_condition':X.Vehicle_condition,
    'type_of_order':X.Type_of_order,
    'type_of_vehicle':X.Type_of_vehicle,
    'multiple_deliveries':X.multiple_deliveries,
    'festival':X.Festival,
    'city_type':X.City
                    }, index=[0])
    
    cleaned_data = perform_data_cleaning(pred_data)
    prediction = model_pipe.predict(cleaned_data)  
    prediction_reshaped = prediction.reshape(-1, 1)
    actual_time = pt.inverse_transform(prediction_reshaped)[0][0]
    return {"predicted_time_taken_minutes": float(actual_time)}


if __name__=="__main__":
    # For Docker runs on linux we need to give host as 0.0.0.0
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=True)

    # For local testing host 127 is fine
    #uvicorn.run(app="app:app", port=8000, reload=True)

# {
#   "ID": "0x4607",
#   "Delivery_person_ID": "INDORES13DEL02",
#   "Delivery_person_Age": "32",
#   "Delivery_person_Ratings": "4.9",
#   "Restaurant_latitude": 22.745049,
#   "Restaurant_longitude": 75.892471,
#   "Delivery_location_latitude": 22.765049,
#   "Delivery_location_longitude": 75.912471,
#   "Order_Date": "19-03-2022",
#   "Time_Orderd": "11:30:00",
#   "Time_Order_picked": "11:50:00",
#   "Weatherconditions": "sunny",
#   "Road_traffic_density": "jam",
#   "Vehicle_condition": 0,
#   "Type_of_order": "snack",
#   "Type_of_vehicle": "motorcycle",
#   "multiple_deliveries": "0",
#   "Festival": "yes",
#   "City": "urban"
# }


    


