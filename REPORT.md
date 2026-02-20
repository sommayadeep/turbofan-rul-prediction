# Turbofan Engine RUL Prediction Report

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
| subset | model_type | rmse | mae |
| --- | --- | --- | --- |
| FD001 | baseline | 15.8144 | 10.1301 |
| FD001 | improved | 15.6636 | 10.0095 |
| FD002 | baseline | 16.7837 | 11.0895 |
| FD002 | improved | 16.7263 | 11.0626 |
| FD003 | baseline | 13.3585 | 7.8894 |
| FD003 | improved | 12.9021 | 7.7372 |
| FD004 | baseline | 15.1072 | 9.1282 |
| FD004 | improved | 14.9326 | 9.2128 |

### 4.2 Baseline vs Improved Summary
| subset | rmse_baseline | rmse_improved | mae_baseline | mae_improved | rmse_gain | mae_gain |
| --- | --- | --- | --- | --- | --- | --- |
| FD001 | 15.8144 | 15.6636 | 10.1301 | 10.0095 | 0.1508 | 0.1206 |
| FD002 | 16.7837 | 16.7263 | 11.0895 | 11.0626 | 0.0574 | 0.0269 |
| FD003 | 13.3585 | 12.9021 | 7.8894 | 7.7372 | 0.4564 | 0.1522 |
| FD004 | 15.1072 | 14.9326 | 9.1282 | 9.2128 | 0.1746 | -0.0846 |

### 4.3 Best Scores
- Best RMSE: `12.9021` on subset `FD003` using `improved`
- Best MAE: `7.7372` on subset `FD003` using `improved`

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
