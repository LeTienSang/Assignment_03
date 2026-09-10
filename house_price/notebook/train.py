from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vietnam_housing_dataset.csv"
MODEL = ROOT / "model"
MODEL.mkdir(exist_ok=True)
NUMERIC = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL = ["District", "House direction", "Balcony direction", "Legal status", "Furniture state"]
FEATURES = NUMERIC + CATEGORICAL


class FiveLayerDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(220, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(16, 1),
        )

    def forward(self, values):
        return self.network(values)


def fixed_features(values, width=220):
    result = np.zeros((values.shape[0], width), dtype=np.float32)
    result[:, :min(values.shape[1], width)] = values[:, :width]
    return result


def regression_metrics(y_true, predictions):
    return {"mae": mean_absolute_error(y_true, predictions), "rmse": np.sqrt(mean_squared_error(y_true, predictions)), "r2": r2_score(y_true, predictions)}

def prepare(df):
    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    for col in NUMERIC: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["District"] = df["Address"].fillna("Unknown").str.extract(r"(?:,\s*)([^,]+)$")[0].fillna("Unknown")
    df["HouseAge"] = 2024 - 2024
    return df

def main():
    df = prepare(pd.read_csv(DATA)).dropna(subset=["Price"])
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df["Price"], test_size=.2, random_state=42)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)])
    x_train_transformed = prep.fit_transform(x_train)
    x_test_transformed = prep.transform(x_test)
    if hasattr(x_train_transformed, "toarray"):
        x_train_transformed = x_train_transformed.toarray()
        x_test_transformed = x_test_transformed.toarray()
    x_train_encoded = fixed_features(x_train_transformed)
    x_test_encoded = fixed_features(x_test_transformed)
    baselines = {
        "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    report = []
    for name, baseline in baselines.items():
        baseline.fit(x_train_encoded, y_train)
        report.append({"model": name, **regression_metrics(y_test, baseline.predict(x_test_encoded))})

    torch.manual_seed(42)
    model = FiveLayerDNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.MSELoss()
    train_x = torch.tensor(x_train_encoded, dtype=torch.float32)
    train_y = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        dl_predictions = model(torch.tensor(x_test_encoded, dtype=torch.float32)).numpy().ravel()
    report.append({"model": "5-Layer DNN", **regression_metrics(y_test, dl_predictions)})

    joblib.dump(prep, MODEL / "preprocessor.joblib")
    torch.save({"state_dict": model.state_dict(), "input_dim": 220, "architecture": [220, 128, 64, 32, 16, 1]}, MODEL / "model_5l.pt")
    pd.DataFrame(report).round(4).to_csv(MODEL / "comparison.csv", index=False)
    print(pd.DataFrame(report).round(3).to_string(index=False))

if __name__ == "__main__": main()
