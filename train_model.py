import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from features_extraction import MODEL_FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset_2.csv"
MODEL_PATH = ROOT / "model.pkl"


def train_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    data = pd.read_csv(dataset_path)
    required_columns = (*MODEL_FEATURE_NAMES, "Label")
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Training dataset is missing columns: {missing_columns}")

    features = data.loc[:, MODEL_FEATURE_NAMES]
    labels = data["Label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)

    with Path(model_path).open("wb") as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    return model, model.score(x_test, y_test)


if __name__ == "__main__":
    trained_model, test_accuracy = train_model()
    print(
        f"Saved {type(trained_model.named_steps['classifier']).__name__} "
        f"pipeline to {MODEL_PATH} (holdout accuracy: {test_accuracy:.4f})"
    )
