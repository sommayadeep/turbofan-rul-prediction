from pathlib import Path

import joblib
import pandas as pd

from pipeline import SUBSETS, train_eval_subset


MODEL_TYPES = ["baseline", "improved"]
RUL_CAP = 125


def main() -> None:
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_type in MODEL_TYPES:
        for subset in SUBSETS:
            result = train_eval_subset(subset=subset, rul_cap=RUL_CAP, model_type=model_type)
            rows.append(
                {
                    "subset": subset,
                    "model_type": model_type,
                    "rul_cap": RUL_CAP,
                    "rmse": round(result["rmse"], 4),
                    "mae": round(result["mae"], 4),
                }
            )

            out_path = models_dir / f"rul_model_{model_type}_{subset}.joblib"
            joblib.dump(
                {
                    "model": result["model"],
                    "features": result["feature_cols"],
                    "rul_cap": RUL_CAP,
                    "model_type": model_type,
                    "subset": subset,
                },
                out_path,
            )
            print(
                f"{subset} | {model_type:<8} | RMSE={result['rmse']:.4f} "
                f"| MAE={result['mae']:.4f} | saved {out_path}"
            )

    results_df = pd.DataFrame(rows).sort_values(["subset", "model_type"]).reset_index(drop=True)
    csv_path = results_dir / "metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    pivot_rmse = results_df.pivot(index="subset", columns="model_type", values="rmse")
    pivot_mae = results_df.pivot(index="subset", columns="model_type", values="mae")

    if set(MODEL_TYPES).issubset(pivot_rmse.columns):
        delta_rmse = pivot_rmse["baseline"] - pivot_rmse["improved"]
        delta_mae = pivot_mae["baseline"] - pivot_mae["improved"]
        summary = pd.DataFrame(
            {
                "rmse_gain_positive_is_better": delta_rmse.round(4),
                "mae_gain_positive_is_better": delta_mae.round(4),
            }
        )
        summary_path = results_dir / "improvement_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved improvement summary to {summary_path}")


if __name__ == "__main__":
    main()
