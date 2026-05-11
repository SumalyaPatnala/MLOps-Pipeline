import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path("artifacts/model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl not found. Please run train.py first."
        )

    return joblib.load(MODEL_PATH)


def run_prediction():
    model = load_model()

    # Example input for Iris dataset
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

    prediction = model.predict(sample_input)

    print("Input:", sample_input)
    print("Prediction:", prediction[0])


if __name__ == "__main__":
    run_prediction()