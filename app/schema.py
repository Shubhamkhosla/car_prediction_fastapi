
from typing import Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict


class CarFeatures(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    brand: str = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1, max_length=50)

    vehicle_age: int = Field(..., ge=0, le=50)
    km_driven: int = Field(..., ge=0, le=1_000_000)

    seller_type: Literal[
        "Individual",
        "Dealer",
        "Trustmark Dealer",
    ]

    fuel_type: Literal[
        "Petrol",
        "Diesel",
        "CNG",
        "LPG",
        "Electric",
    ]

    transmission_type: Literal["Manual", "Automatic"]

    mileage: float = Field(..., gt=0, le=100)
    engine: float = Field(..., gt=0, le=10_000)
    max_power: float = Field(..., gt=0, le=2_000)
    seats: int = Field(..., ge=1, le=20)

    @field_validator("brand", "model")
    @classmethod
    def validate_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Brand and model cannot be empty")

        return value

    @field_validator("mileage", "engine", "max_power")
    @classmethod
    def validate_positive_numbers(cls, value: float) -> float:
        if not value.is_integer() and value < 0:
            raise ValueError("Value must be positive")

        return value
