import pandas as pd 
import numpy as np
import os
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient
import io

load_dotenv()

# Azure Blob Connection Details
conn_str = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_INPUT_CONTAINER_NAME")
blob_name = os.getenv("AZURE_INPUT_BLOB_PREFIX")

blob_service_client = BlobServiceClient.from_connection_string(conn_str)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Read csv file from Azure Blob Storage
stream = blob_client.download_blob().readall()
df = pd.read_csv(io.BytesIO(stream))


# Make directory for storing data locally
data_path = os.path.join("data","raw")
os.makedirs(data_path, exist_ok=True)

# Clean column names
def change_column_names(data : pd.DataFrame):
  data.columns = data.columns.str.lower()
  return data.rename(columns={
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"
            })


cleaned_data = df.pipe(change_column_names)

cleaned_data.to_csv(os.path.join(data_path,"swiggy_input.csv"),index=False)







