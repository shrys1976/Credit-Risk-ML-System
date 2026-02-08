import joblib
import os


def save_model(model_pipeline, model_name="xgb_credit_model"):

    os.makedirs("artifacts", exist_ok=True)

    model_path = f"artifacts/{model_name}.joblib"

    joblib.dump(model_pipeline, model_path)

    return model_path
