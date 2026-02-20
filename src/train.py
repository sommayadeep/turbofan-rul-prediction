import argparse
from pathlib import Path

import joblib

from pipeline import SUBSETS, train_eval_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RUL model for C-MAPSS")
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--model-type", default="baseline", choices=["baseline", "improved"])
    args = parser.parse_args()

    result = train_eval_subset(subset=args.subset, rul_cap=args.rul_cap, model_type=args.model_type)

    print(f"Subset: {result['subset']}")
    print(f"Model:  {result['model_type']}")
    print(f"RUL cap: {result['rul_cap']}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"MAE:  {result['mae']:.4f}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"rul_model_{args.model_type}_{args.subset}.joblib"
    joblib.dump(
        {
            "model": result["model"],
            "features": result["feature_cols"],
            "rul_cap": args.rul_cap,
            "model_type": args.model_type,
            "subset": args.subset,
        },
        out_path,
    )
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
