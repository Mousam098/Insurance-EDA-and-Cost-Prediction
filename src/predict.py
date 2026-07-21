"""Example inference script for the trained insurance model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "insurance_cost_model.joblib"


def predict_charge(age, sex, bmi, children, smoker, region):
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])
    return float(model.predict(sample)[0])


if __name__ == "__main__":
    value = predict_charge(35, "male", 28.5, 1, "no", "southeast")
    print(f"Predicted insurance charge: ${value:,.2f}")
