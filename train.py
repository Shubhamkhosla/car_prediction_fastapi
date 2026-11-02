
import json
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cardekho_dataset.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

target_column = "selling_price"

required_columns = [
    "selling_price",
    "car_name",
    "model",
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "brand",
    "seller_type",
    "fuel_type",
    "transmission_type",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# Keep only the columns used by this training script
df = df[required_columns].copy()

# Convert numeric columns to numeric types
num_features = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

cat_features = [
    "brand",
    "seller_type",
    "fuel_type",
    "transmission_type",
]

for column in num_features + [target_column]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing target values
df = df.dropna(subset=[target_column])

# Impute missing feature values using training data later
X = df.drop(columns=["selling_price", "car_name", "model"])
y = df[target_column]

# Remove rows where all feature values are missing
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# 4. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

# Numeric missing values are filled using training medians.
# Categorical missing values are filled using training modes.
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_features,
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            cat_features,
        ),
    ],
)


# --------------------------------------------------
# 6. Build and train pipeline
# --------------------------------------------------

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
    ),
])

print("\nTraining model...")
xgb_pipeline.fit(X_train, y_train)
print("Training completed!")


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

y_pred = xgb_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"R2 Score: {r2:.4f}")
print(f"MAE:      {mae:.2f}")
print(f"MSE:      {mse:.2f}")


# --------------------------------------------------
# 8. Save model artifacts for FastAPI
# --------------------------------------------------

# Save the fitted preprocessing pipeline separately
joblib.dump(
    xgb_pipeline.named_steps["preprocessor"],
    BASE_DIR / "preprocessor.pkl",
)

# Save the trained XGBoost regressor in native JSON format
xgb_pipeline.named_steps["regressor"].save_model(
    str(BASE_DIR / "xgb_model.json"),
)

# Save the feature metadata
feature_metadata = {
    "numerical_features": num_features,
    "categorical_features": cat_features,
    "input_features": X_train.columns.tolist(),
}

with open(
    BASE_DIR / "feature_columns.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(feature_metadata, file, indent=2)

print("\nModel artifacts saved successfully:")
print(BASE_DIR / "xgb_model.json")
print(BASE_DIR / "preprocessor.pkl")
print(BASE_DIR / "feature_columns.json")
