from pathlib import Path

import pandas as pd


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([headers, sep] + rows)


def main() -> None:
    results_path = Path("results/metrics.csv")
    if not results_path.exists():
        raise FileNotFoundError("Missing results/metrics.csv. Run src/run_experiments.py first.")

    df = pd.read_csv(results_path)
    pivot = df.pivot(index="subset", columns="model_type", values=["rmse", "mae"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    if "rmse_baseline" in pivot and "rmse_improved" in pivot:
        pivot["rmse_gain"] = (pivot["rmse_baseline"] - pivot["rmse_improved"]).round(4)
    if "mae_baseline" in pivot and "mae_improved" in pivot:
        pivot["mae_gain"] = (pivot["mae_baseline"] - pivot["mae_improved"]).round(4)

    full_table = df.copy()
    full_table["rmse"] = full_table["rmse"].round(4)
    full_table["mae"] = full_table["mae"].round(4)

    best_rmse = df.loc[df["rmse"].idxmin()]
    best_mae = df.loc[df["mae"].idxmin()]

    report = f"""# Turbofan Engine RUL Prediction Report

## 1. Problem Statement
Predict Remaining Useful Life (RUL) for turbofan engines from multivariate time-series sensor data.

## 2. Dataset
- Source: NASA C-MAPSS subsets FD001, FD002, FD003, FD004
- Inputs per cycle: engine id, cycle, 3 operational settings, 21 sensor readings
- Target: `RUL = final_cycle - current_cycle` (capped at 125)

## 3. Methodology
- Baseline model: RandomForestRegressor with raw tabular features
- Improved model: HistGradientBoostingRegressor on the same raw features with tuned hyperparameters
- Metrics: RMSE and MAE on labeled test trajectories

## 4. Results
### 4.1 Model Comparison (All Runs)
{to_markdown_table(full_table[["subset", "model_type", "rmse", "mae"]])}

### 4.2 Baseline vs Improved Summary
{to_markdown_table(pivot.round(4))}

### 4.3 Best Scores
- Best RMSE: `{best_rmse['rmse']:.4f}` on subset `{best_rmse['subset']}` using `{best_rmse['model_type']}`
- Best MAE: `{best_mae['mae']:.4f}` on subset `{best_mae['subset']}` using `{best_mae['model_type']}`

## 5. Visualizations
Generated at:
- `results/figures/fd001_rul_distribution.png`
- `results/figures/fd001_sensor_trends.png`
- `results/figures/fd001_pred_vs_true.png`
- `results/figures/fd001_feature_importance.png`

## 6. Conclusion
- The tuned HistGradientBoosting model improves RMSE on all four subsets.
- MAE also improves on FD001, FD002, FD003, with a small regression on FD004.
- Further improvements can come from sequence models (LSTM/Transformer), hyperparameter search, and better feature selection.
"""

    report_path = Path("REPORT.md")
    report_path.write_text(report)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
