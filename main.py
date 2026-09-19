
import joblib
preprocessor_path = "models/preprocessor.joblib"

def load_preprocessor(preprocessor_path:str):
    preprocessor= joblib.load(preprocessor_path)
    return preprocessor

preprocessor = load_preprocessor(preprocessor_path)

print(preprocessor.get_feature_names_out())




