# Project 6: Turbofan Engine RUL Prediction

Predict Remaining Useful Life (RUL), i.e. the number of cycles left before failure, using NASA C-MAPSS data.

## Objective
Given multivariate sensor time series for each engine, estimate:

`RUL = final_cycle - current_cycle`

## Dataset
Use the C-MAPSS subsets in `data/raw/`:
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

Each row has: `engine_id`, `cycle`, `os_1..os_3`, and 21 sensor values.

## Project Structure
- `src/pipeline.py`: shared data loading, target creation, feature building, model factory
- `src/train.py`: train/evaluate one subset (`baseline` or `improved`)
- `src/run_experiments.py`: run all subsets for both model types and save metrics
- `src/eda.py`: generate plots
- `src/generate_report.py`: generate `REPORT.md`
- `results/metrics.csv`: all metrics
- `results/improvement_summary.csv`: baseline vs improved gains
- `results/figures/`: generated visualizations
- `models/`: saved trained models

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --default-timeout=120 --retries 8 -r requirements.txt
```

## Run One Model
```bash
python src/train.py --subset FD001 --model-type baseline --rul-cap 125
python src/train.py --subset FD001 --model-type improved --rul-cap 125
```

## Run Full Pipeline (Recommended)
```bash
python src/run_experiments.py
python src/eda.py
python src/generate_report.py
```

## Final Results (RUL cap = 125)

| Subset | Baseline RMSE | Improved RMSE | Baseline MAE | Improved MAE |
|---|---:|---:|---:|---:|
| FD001 | 15.8144 | 15.6636 | 10.1301 | 10.0095 |
| FD002 | 16.7837 | 16.7263 | 11.0895 | 11.0626 |
| FD003 | 13.3585 | 12.9021 | 7.8894 | 7.7372 |
| FD004 | 15.1072 | 14.9326 | 9.1282 | 9.2128 |

## Models
- `baseline`: `RandomForestRegressor`
- `improved`: `HistGradientBoostingRegressor` (tuned)

## Internship Submission Files
Use these as your final submission package:
- `README.md`
- `REPORT.md`
- `results/metrics.csv`
- `results/improvement_summary.csv`
- `results/figures/*.png`
- `src/*.py`
