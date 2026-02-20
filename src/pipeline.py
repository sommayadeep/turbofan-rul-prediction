from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_COLS = ["engine_id", "cycle", "os_1", "os_2", "os_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = BASE_COLS + SENSOR_COLS
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


@dataclass
class DatasetBundle:
    subset: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def read_cmapss_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] > len(ALL_COLS):
        df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def add_rul_targets(train_df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    max_cycle = train_df.groupby("engine_id")["cycle"].transform("max")
    out = train_df.copy()
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=rul_cap)
    return out


def build_test_targets(test_df: pd.DataFrame, rul_file: Path, rul_cap: int) -> pd.DataFrame:
    rul_offsets = pd.read_csv(rul_file, sep=r"\s+", header=None).iloc[:, 0].values
    last_cycle = test_df.groupby("engine_id")["cycle"].max().sort_index()
    engine_ids = last_cycle.index.values

    if len(engine_ids) != len(rul_offsets):
        raise ValueError("Mismatch between number of test engines and RUL rows")

    final_cycle = pd.Series(last_cycle.values + rul_offsets, index=engine_ids)
    out = test_df.copy()
    out["final_cycle"] = out["engine_id"].map(final_cycle)
    out["rul"] = (out["final_cycle"] - out["cycle"]).clip(upper=rul_cap)
    return out.drop(columns=["final_cycle"])


def load_subset(subset: str, rul_cap: int, data_dir: Path = Path("data/raw")) -> DatasetBundle:
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    train_df = add_rul_targets(read_cmapss_file(train_path), rul_cap=rul_cap)
    test_df = build_test_targets(read_cmapss_file(test_path), rul_file=rul_path, rul_cap=rul_cap)
    return DatasetBundle(subset=subset, train_df=train_df, test_df=test_df)


def add_engineered_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("engine_id", group_keys=False)

    out["cycle_norm"] = out["cycle"] / g["cycle"].transform("max")

    for col in SENSOR_COLS:
        out[f"{col}_diff1"] = g[col].diff().fillna(0.0)
        out[f"{col}_roll_mean_{window}"] = (
            g[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"{col}_roll_std_{window}"] = (
            g[col].rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    return out


def build_features(df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, List[str]]:
    if model_type == "baseline":
        feature_cols = ["cycle", "os_1", "os_2", "os_3"] + SENSOR_COLS
        return df[feature_cols], feature_cols

    if model_type == "improved":
        feature_cols = ["cycle", "os_1", "os_2", "os_3"] + SENSOR_COLS
        return df[feature_cols], feature_cols

    raise ValueError(f"Unknown model type: {model_type}")


def make_model(model_type: str, random_state: int = 42):
    if model_type == "baseline":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "improved":
        return HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def train_eval_subset(subset: str, rul_cap: int, model_type: str) -> Dict[str, float]:
    data = load_subset(subset=subset, rul_cap=rul_cap)

    x_train, feature_cols = build_features(data.train_df, model_type=model_type)
    x_test, _ = build_features(data.test_df, model_type=model_type)
    y_train = data.train_df["rul"]
    y_test = data.test_df["rul"]

    model = make_model(model_type=model_type)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))

    return {
        "subset": subset,
        "model_type": model_type,
        "rul_cap": rul_cap,
        "rmse": rmse,
        "mae": mae,
        "model": model,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "pred": pred,
    }
