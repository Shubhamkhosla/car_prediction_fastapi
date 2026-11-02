
import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

CAR_API_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = CAR_API_DIR / "xgb_model.json"
PREPROCESSOR_PATH = CAR_API_DIR / "preprocessor.pkl"
FEATURE_PATH = CAR_API_DIR / "feature_columns.json"

model = None
preprocessor = None
feature_metadata = None


def load_artifacts():
    global model, preprocessor, feature_metadata

    if model is None:
        model = XGBRegressor()
        model.load_model(str(MODEL_PATH))

    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    if feature_metadata is None:
        with open(FEATURE_PATH, "r", encoding="utf-8") as f:
            feature_metadata = json.load(f)

    return model, preprocessor, feature_metadata



def predict_prices(data: list[dict]) -> list[float]:

    model, preprocessor, metadata = load_artifacts()

    required = metadata["input_features"]

    input_df = pd.DataFrame([
        {col: car[col] for col in required}
        for car in data
    ])

    transformed = preprocessor.transform(input_df)

    predictions = model.predict(transformed)

    return [float(price) for price in predictions]
