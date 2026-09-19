import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, PowerTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pickle, joblib
import yaml
from typing import Union
import logging

# logging configure
logger = logging.getLogger('feature_enngineering_logger')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# Here input params_path is string and output test size is float
def load_params(params_path: str) -> float:
    """
    Loads the test_size parameter from the feature engineering section of a YAML file.

    Args:
        params_path (str): Path to the parameters YAML file.

    Returns:
        float: The test_size value.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
        KeyError: If the required keys are missing in the YAML file.
        ValueError: If test_size is not a float.
    """
    try:
        logger.info(f"Attempting to load parameters from '{params_path}'")
        # Open and read the YAML file
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info("Successfully read the YAML file.")    
        
        # Access the required value
        test_size = params['feature_engineering']['test_size']
        logger.info(f"Fetched 'test_size': {test_size}")
        
        # Validate the type of test_size
        if not isinstance(test_size, (float, int)):
            raise ValueError(f"Expected test_size to be a float or int, but got {type(test_size)}")
        
        logger.info("Validation successful. Returning test_size.")
        return float(test_size)  # Ensure the returned value is a float

    except FileNotFoundError:
        logger.error(f"The file at path '{params_path}' was not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing required key in YAML file: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except ValueError as e:
        logger.error(e)
        raise

def read_data(url:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        logger.error("File Not found")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    # drop columns not required for model input
    columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_hour",
                    "city_name","order_month"]
    
    # Drop missing values also
    temp_df = df.drop(labels=columns_to_drop, axis=1, errors='ignore').dropna()
    return temp_df

def save_transformer(transfomer, save_dir: Path, transformer_name:str):
    save_location = save_dir / transformer_name 
    joblib.dump(value=transfomer, filename = save_location)




def split_data(temp_df: pd.DataFrame,test_size:float)->tuple:
    # split into X and y
    X = temp_df.drop(columns='time_taken')
    y = temp_df['time_taken']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    return X_train, X_test, y_train, y_test


def data_preprocessing(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # do basic preprocessing
    num_cols = ["age","ratings","pick_up_time_mins","distance"]

    nominal_cat_cols = ['weather','type_of_order',
                    'type_of_vehicle',"festival",
                    "city_type",
                    "is_weekend",
                    "time_of_day"]

    ordinal_cat_cols = ["traffic","distance_type"]


    # generate order for ordinal encoding
    traffic_order = ["low","medium","high","jam"]
    distance_type_order = ["short","medium","long","very_long"]

    # build a preprocessor

    preprocessor = ColumnTransformer(transformers=[
    ("scale", MinMaxScaler(), num_cols),
    ("nominal_encode", OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False), nominal_cat_cols),
    ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,distance_type_order], 
                                                  encoded_missing_value=-999,
                                                  handle_unknown="use_encoded_value",
                                                  unknown_value=-1),ordinal_cat_cols)
                                                 ],
                                                 remainder="passthrough",
                                                 verbose_feature_names_out=False)

    preprocessor.set_output(transform="pandas")

    # transform the data
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # transform target column
    pt = PowerTransformer()
    y_train_pt = pt.fit_transform(y_train.values.reshape(-1,1))
    y_test_pt = pt.transform(y_test.values.reshape(-1,1))

    # Combine X and Y into dataframe to store entire data 
    train_features = pd.DataFrame(X_train_trans)
    train_features['time_taken'] = y_train_pt

    test_features = pd.DataFrame(X_test_trans)
    test_features['time_taken'] = y_test_pt

    return pt,train_features, test_features,preprocessor


def save_data(data_path,pt,train_features, test_features,X_train,X_test,y_train,y_test)->None:
    # Create folder
    data_path.mkdir(exist_ok=True,parents=True)

    # Write transformed features to csv
    train_features.to_csv(data_path / "train_features.csv",index=False)
    test_features.to_csv(data_path / "test_features.csv",index=False)

    # Write original features to csv
    train = pd.DataFrame(X_train)
    train['time_taken'] = y_train

    test = pd.DataFrame(X_test)
    test['time_taken'] = y_test

    train.to_csv(data_path / "train.csv",index=False)
    test.to_csv(data_path / "test.csv",index=False)

    # Save Power Transformer 
    pickle.dump(pt,open("power_transformer.pkl","wb"))


ROOT_FOLDER = Path(__file__).resolve().parent.parent
INPUT_FOLDER = ROOT_FOLDER / "data" / "processed" 
OUTPUT_FOLDER = ROOT_FOLDER / "data" / "features"
transformer_save_dir = ROOT_FOLDER / "models" 
transformer_filename = "preprocessor.joblib"


def main():
    test_size = load_params("params.yaml")
    file_path = INPUT_FOLDER / "swiggy_cleaned.csv"
    df = read_data(file_path.as_posix())
    temp_df = drop_columns(df)
    X_train, X_test, y_train, y_test = split_data(temp_df,test_size)
    pt,train_features, test_features,preprocessor = data_preprocessing(X_train, X_test, y_train, y_test)
    save_data(OUTPUT_FOLDER ,pt,train_features, test_features,X_train,X_test,y_train,y_test)
    transformer_save_dir.mkdir(exist_ok=True,parents=True)
    save_transformer(transfomer=preprocessor, 
                     save_dir = transformer_save_dir, 
                     transformer_name=transformer_filename)
    logger.info("Preprocessor saved to location")

if __name__=="__main__":
    main()    

    
