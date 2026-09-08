"""Train (once) and load the iris classifier."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "iris_model.joblib"


def train_and_save_model() -> None:
    data = load_iris()
    clf = LogisticRegression(max_iter=200)
    clf.fit(data.data, data.target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "target_names": data.target_names.tolist()}, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)


def predict_species(model_bundle: dict, feature_vector: list[float]) -> str:
    model = model_bundle["model"]
    target_names = model_bundle["target_names"]
    pred_idx = model.predict([feature_vector])[0]
    return target_names[pred_idx]