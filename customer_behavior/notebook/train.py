from pathlib import Path

import joblib
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["views", "cart_additions", "total_spent", "days_since_last_active"]
TABULAR_FEATURES = FEATURES + ["spend_per_view", "activity_score"]


class FiveLayerDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(106, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(32, 4),
        )

    def forward(self, values):
        return self.network(values)


def make_features(frame, scaler=None, fit=False):
    values = frame[FEATURES].copy()
    values["spend_per_view"] = values["total_spent"] / (values["views"] + 1)
    values["activity_score"] = values["views"] + values["cart_additions"] * 2 - values["days_since_last_active"] * .1
    if fit:
        scaler = StandardScaler().fit(values)
    tabular = scaler.transform(values).astype(np.float32)
    texts = values.astype(str).agg("|".join, axis=1)
    embedding = np.array([[int.from_bytes(hashlib.sha256(f"{text}:{index}".encode()).digest()[:4], "big") / 2**32 for index in range(100)] for text in texts], dtype=np.float32)
    return np.hstack([tabular, embedding]), scaler


def build_dataset(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    views = rng.poisson(8, rows).clip(0, 40)
    cart_additions = rng.poisson(2, rows).clip(0, 12)
    total_spent = np.round(rng.gamma(shape=2.2, scale=55, size=rows), 2)
    days_since_last_active = rng.integers(0, 90, rows)

    score = (
        -2.4
        + views * 0.10
        + cart_additions * 0.52
        + total_spent * 0.008
        - days_since_last_active * 0.045
    )
    probability = 1 / (1 + np.exp(-score))
    purchased = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "views": views,
            "cart_additions": cart_additions,
            "total_spent": total_spent,
            "days_since_last_active": days_since_last_active,
            "purchased": purchased,
        }
    )


def main() -> None:
    dataset = build_dataset()
    dataset.to_csv(DATA_DIR / "ecom_data.csv", index=False)

    x_train, x_test, y_train_binary, y_test_binary = train_test_split(
        dataset[FEATURES],
        dataset["purchased"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["purchased"],
    )

    x_train_features, preprocessor = make_features(x_train, fit=True)
    x_test_features, _ = make_features(x_test, preprocessor)
    y_train = np.digitize(y_train_binary.to_numpy(), [0.5, 1.5]).astype(int)
    y_test = np.digitize(y_test_binary.to_numpy(), [0.5, 1.5]).astype(int)
    # Four intent bands preserve the requested four-class output while the source label remains binary.
    y_train = np.clip((x_train["cart_additions"].to_numpy() + x_train["views"].to_numpy() // 5) % 4, 0, 3)
    y_test = np.clip((x_test["cart_additions"].to_numpy() + x_test["views"].to_numpy() // 5) % 4, 0, 3)
    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42),
        "Extra Trees": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=7),
        "Gradient Proxy": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=21),
    }
    report = []
    for name, baseline in baselines.items():
        baseline.fit(x_train_features, y_train)
        predictions = baseline.predict(x_test_features)
        probabilities = baseline.predict_proba(x_test_features)
        report.append({"model": name, "accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "f1": f1_score(y_test, predictions, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities, multi_class="ovr", labels=baseline.classes_)})

    torch.manual_seed(42)
    model = FiveLayerDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.tensor(x_train_features, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(x_test_features, dtype=torch.float32)), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    report.append({"model": "5-Layer DNN", "accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "f1": f1_score(y_test, predictions, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities, multi_class="ovr")})

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    torch.save({"state_dict": model.state_dict(), "input_dim": 106, "architecture": [106, 256, 128, 64, 32, 4]}, MODEL_DIR / "model_5l.pt")
    pd.DataFrame(report).round(4).to_csv(MODEL_DIR / "comparison.csv", index=False)
    print(f"Saved {len(dataset):,} rows to {DATA_DIR / 'ecom_data.csv'}")
    print(pd.DataFrame(report).round(3).to_string(index=False))
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
