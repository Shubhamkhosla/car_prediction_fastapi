
from fastapi import FastAPI, HTTPException
from app.model import predict_prices
from app.schema import CarFeatures

app = FastAPI(title="Car Price Prediction API")


@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}


@app.post("/predict")
def predict(data: list[CarFeatures]):

    try:
        # Convert list of Pydantic objects into dictionaries
        cars = [car.model_dump() for car in data]

        # Generate predictions for all cars together
        prices = predict_prices(cars)

        results = []

        for car, price in zip(cars, prices):
            results.append({
                "brand": car["brand"],
                "model": car["model"],
                "predicted_price": round(float(price), 2),
                "currency": "INR"
            })

        return {
            "total_cars": len(results),
            "predictions": results
        }

    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("Prediction error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
