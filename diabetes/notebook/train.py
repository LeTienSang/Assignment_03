from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/diabetes.csv"
MODEL = ROOT / "model"
MODEL.mkdir(exist_ok=True)
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


class FiveLayerDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, values):
        return self.network(values)


def metrics(y_true, probabilities):
    predictions = (probabilities >= .5).astype(int)
    return {"accuracy": accuracy_score(y_true, predictions), "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0), "f1": f1_score(y_true, predictions, zero_division=0), "roc_auc": roc_auc_score(y_true, probabilities)}

def main():
    df = pd.read_csv(DATA)
    df[FEATURES] = df[FEATURES].replace(0, np.nan)
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df["Outcome"], test_size=.2, random_state=42, stratify=df["Outcome"])
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)])
    x_train_scaled = prep.fit_transform(x_train)
    x_test_scaled = prep.transform(x_test)
    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    report = []
    for name, baseline in baselines.items():
        baseline.fit(x_train_scaled, y_train)
        report.append({"model": name, **metrics(y_test, baseline.predict_proba(x_test_scaled)[:, 1])})

    torch.manual_seed(42)
    model = FiveLayerDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.BCELoss()
    train_x = torch.tensor(x_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        dl_probabilities = model(torch.tensor(x_test_scaled, dtype=torch.float32)).numpy().ravel()
    report.append({"model": "5-Layer DNN", **metrics(y_test, dl_probabilities)})

    joblib.dump(prep, MODEL / "preprocessor.joblib")
    torch.save({"state_dict": model.state_dict(), "input_dim": 8, "architecture": [8, 64, 32, 16, 8, 1]}, MODEL / "model_5l.pt")
    pd.DataFrame(report).round(4).to_csv(MODEL / "comparison.csv", index=False)
    print(pd.DataFrame(report).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
