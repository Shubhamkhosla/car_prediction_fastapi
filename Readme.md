# 🚗 Used Car Price Prediction

An end-to-end Machine Learning application that predicts the selling price of used cars using **XGBoost**. The project uses FastAPI to serve predictions through a REST API and Streamlit to provide an interactive user interface.

## 🌟 Features

- Predict used car selling prices using an XGBoost regression model.
- Predict prices for multiple cars in a single request.
- Interactive Streamlit user interface.
- FastAPI REST API for model predictions.
- Data preprocessing using Scikit-learn.
- Categorical feature encoding using OneHotEncoder.
- Numerical feature scaling using StandardScaler.
- Pydantic validation for API inputs.
- Deployment using Render.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing |
| XGBoost | Machine Learning model |
| FastAPI | Backend REST API |
| Pydantic | Input validation |
| Streamlit | Frontend interface |
| Joblib | Saving and loading preprocessing objects |
| Render | Cloud deployment |

## 📂 Project Structure

```text
car_fastapi/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── schema.py
│
├── cardekho_dataset.csv
├── train.py
├── run.py
├── streamlit_app.py
│
├── xgb_model.json
├── preprocessor.pkl
├── feature_columns.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset and Features

The project uses a used-car dataset containing information about vehicles.

### Input Features

- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Seller Type
- Fuel Type
- Transmission Type
- Mileage
- Engine Capacity
- Maximum Power
- Number of Seats

**Target Variable:** `selling_price`

The `model` and `car_name` columns are excluded from the model's training features in the current training pipeline.

## 🤖 Machine Learning Model

The project uses **XGBoost Regressor (Extreme Gradient Boosting)** to predict used car prices.

### Data Preprocessing

- Numerical features are processed using StandardScaler.
- Categorical features are transformed using OneHotEncoder.
- Unknown categorical values are handled using `handle_unknown="ignore"`.

### Model Configuration

```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
```

### Model Performance

**Test R² Score: 0.9174**

The model achieved an R² score of approximately 0.9174 on the test dataset, meaning it explains approximately 91.74% of the variation in the test-set selling prices relative to a mean-baseline model.

*Note: R² is not the percentage of predictions that are correct.*

## 💾 Saved Model Files

The trained model and preprocessing information are saved separately:

- `xgb_model.json` – Trained XGBoost model.
- `preprocessor.pkl` – Fitted Scikit-learn preprocessing pipeline.
- `feature_columns.json` – Feature names and metadata required for prediction.

## ⚙️ Installation and Local Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd car_fastapi
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root directory, run:

```bash
python -m uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

The Streamlit application will open in your browser.

## 🔌 API Usage

### Endpoint

```text
POST /predict
```

### Example Request

The API accepts a list of car objects, allowing multiple predictions in one request.

```json
[
  {
    "brand": "Maruti",
    "model": "Alto",
    "vehicle_age": 5,
    "km_driven": 45000,
    "seller_type": "Individual",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 20.5,
    "engine": 998,
    "max_power": 67.1,
    "seats": 5
  }
]
```

### Example Response

```json
{
  "total_cars": 1,
  "predictions": [
    {
      "brand": "Maruti",
      "model": "Alto",
      "predicted_price": 350000.0,
      "currency": "INR"
    }
  ]
}
```

*The predicted price above is illustrative, not a guaranteed model output.*

## ☁️ Deployment on Render

The application uses two separate Render Web Services.

### 1. FastAPI Backend

- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 2. Streamlit Frontend

- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

### Configure Environment Variable

In the Streamlit service on Render, add:

| Key = Value |
 `API_URL= `https://YOUR-FASTAPI-URL.onrender.com/predict`

This connects the Streamlit frontend to the deployed FastAPI backend.

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Add more vehicle-related features.
- Implement model evaluation using MAE and RMSE.
- Add visualizations for predicted prices.
- Improve input validation and error handling.
- Explore additional regression algorithms for comparison.


---

⭐ If you find this project useful, consider giving the repository a star!
