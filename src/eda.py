from pathlib import path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline import SENSOR_COLS, load_subset, train_eval_subset


RUL_CAP = 125
SUBSET = "FD001"


def plot_rul_distribution(train_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(train_df["rul"], bins=35, kde=True, color="#1f77b4")
    plt.title("Train RUL Distribution (FD001)")
    plt.xlabel("RUL")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_sensor_trends(train_df: pd.DataFrame, out_path: Path) -> None:
    sensors = ["s_2", "s_7", "s_12"]
    engines = train_df["engine_id"].drop_duplicates().head(4).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False)
    for ax, sensor in zip(axes, sensors):
        for eng_id in engines:
            df_e = train_df[train_df["engine_id"] == eng_id]
            ax.plot(df_e["cycle"], df_e[sensor], linewidth=1.1, label=f"E{eng_id}")
        ax.set_title(f"{sensor} vs Cycle")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensor Trajectories for Sample Engines (FD001)", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pred_vs_true(y_true: pd.Series, y_pred, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=9, alpha=0.5, color="#2a9d8f")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.title("Predicted vs True RUL (FD001, Improved Model)")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_feature_importance(subset: str, out_path: Path) -> None:
    # Use baseline RF for native impurity-based feature importance plot.
    result = train_eval_subset(subset=subset, rul_cap=RUL_CAP, model_type="baseline")
    importances = result["model"].feature_importances_
    feat = pd.DataFrame({"feature": result["feature_cols"], "importance": importances})
    feat = feat.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat, x="importance", y="feature", color="#e76f51")
    plt.title("Top 20 Feature Importances (FD001, Baseline RF)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    figs_dir = Path("results/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    data = load_subset(SUBSET, rul_cap=RUL_CAP)
    plot_rul_distribution(data.train_df, figs_dir / "fd001_rul_distribution.png")
    plot_sensor_trends(data.train_df, figs_dir / "fd001_sensor_trends.png")

    pred_result = train_eval_subset(subset=SUBSET, rul_cap=RUL_CAP, model_type="improved")
    plot_pred_vs_true(pred_result["y_test"], pred_result["pred"], figs_dir / "fd001_pred_vs_true.png")
    plot_feature_importance(SUBSET, figs_dir / "fd001_feature_importance.png")

    print("Saved figures to results/figures")


if __name__ == "__main__":
    main()
