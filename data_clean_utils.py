import pandas as pd
import numpy as np

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

def clean_lat_long(data : pd.DataFrame, threshold=1):
  cols = ['restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude']
  for col in cols:
    data[col] = np.where(data[col]<threshold, np.nan,data[col].values)
  return data


def clean_data(data: pd.DataFrame):
    data = data.drop('id', axis=1)  # drop id column
    minor_df = data[data['age'].astype('float') < 19]
    minor_index = minor_df.index.tolist()
    six_rating_index = data[data['ratings'].astype('float') > 5].index.tolist()

    data['city_name'] = data['rider_id'].str.split("RES").str.get(0)  # taking out city name
    data.drop(index=minor_index, inplace=True)  # removing minor data
    data.drop(index=six_rating_index, inplace=True)  # removing six star data
    data.replace("NaN ", np.nan, inplace=True)  # replace NaN string with np.nan

    data['age'] = data['age'].astype('float')  # Change data type from object to float for age column
    data['ratings'] = data['ratings'].astype('float')  # Change data type from object to float for ratings column
    data['restaurant_latitude'] = data['restaurant_latitude'].abs()  # Change -ve values to +ve fpr lat long
    data['restaurant_longitude'] = data['restaurant_longitude'].abs()  # Change -ve values to +ve fpr lat long
    data['delivery_latitude'] = data['delivery_latitude'].abs()  # Change -ve values to +ve fpr lat long
    data['delivery_longitude'] = data['delivery_longitude'].abs()  # Change -ve values to +ve fpr lat long

    data['order_date'] = pd.to_datetime(data['order_date'], format="%d-%m-%Y", errors='coerce')
    data['order_month'] = data['order_date'].dt.month
    data['is_weekend'] = data['order_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(
        'int')  # If ordered on weekend or not

    data['order_time'] = pd.to_datetime(data['order_time'], format='mixed', errors='coerce')
    data['order_picked_time'] = pd.to_datetime(data['order_picked_time'], format='mixed', errors='coerce')
    data['order_hour'] = data['order_time'].dt.hour
    # What time of day was ordered
    data['time_of_day'] = np.select(condlist=[
        (data['order_hour'].between(6, 12, inclusive="left")),
        (data['order_hour'].between(12, 17, inclusive="left")),
        (data['order_hour'].between(17, 20, inclusive="left")),
        (data['order_hour'].between(20, 24, inclusive="left"))],
        choicelist=["morning", "afternoon", "evening", "night"],
        default="after_midnight")

    data['pick_up_time_mins'] = (data['order_picked_time'] - data['order_time']).dt.seconds / 60
    # Clean weather column and replace nan string with np.nan
    data['weather'] = data["weather"].str.replace("conditions ", "").str.lower().replace("nan", np.nan)
    # Remove right space from traffic
    data['traffic'] = data['traffic'].str.rstrip().str.lower()
    data['type_of_order'] = data['type_of_order'].str.rstrip().str.lower()
    data['type_of_vehicle'] = data['type_of_vehicle'].str.rstrip().str.lower()
    data['multiple_deliveries'] = data['multiple_deliveries'].astype("float")
    data['festival'] = data['festival'].str.rstrip().str.lower()
    data['city_type'] = data['city_type'].str.rstrip().str.lower()
    data['time_taken'] = data['time_taken'].str.replace("(min) ", "").astype('int')
    data.drop(['order_time', 'order_picked_time'], axis=1, inplace=True)

    return data


def calculate_haversine_distance(df):
    location_columns = ['restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude']
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )


def perform_data_cleaning(data:pd.DataFrame, saved_data_path="swiggy_cleaned.csv"):
    cleaned_data = data.pipe(change_column_names).\
               pipe(clean_data).\
               pipe(clean_lat_long).\
               pipe(calculate_haversine_distance)

    cleaned_data.to_csv(saved_data_path, index=False)


if __name__ == '__main__':
    data_path="swiggy.csv"
    df = pd.read_csv(data_path)
    print("Data successfully loaded")

    perform_data_cleaning(df)




