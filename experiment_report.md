# Experiment Report

Generated: 2026-03-16 18:51:00

## Summary

- Total experiments: 9
- Successful: 9
- Failed: 0

## Results

| Model | Parameters | RMSE | MAE | Run ID |
|-------|------------|------|-----|--------|
| svd | {"lr_all": 0.005, "n_epochs": 20, "n_factors": 50, "reg_all": 0.02} | 0.9346 | 0.7376 | 84f9d9a208dc4a72bc459078b47f7112 |
| svd | {"lr_all": 0.005, "n_epochs": 20, "n_factors": 100, "reg_all": 0.02} | 0.9357 | 0.7376 | eafbde21def745a5801e767e03264692 |
| svd | {"lr_all": 0.01, "n_epochs": 30, "n_factors": 150, "reg_all": 0.02} | 0.9600 | 0.7517 | f97b007a1d6449b18d9b56659338a802 |
| nmf | {"n_epochs": 30, "n_factors": 30} | 1.1802 | 0.9002 | a69e454dcac14a29b42964bb8691244e |
| nmf | {"n_epochs": 50, "n_factors": 50} | 1.0303 | 0.7842 | 3f4699a02e47421f88aa2cf7b46da9ff |
| nmf | {"n_epochs": 70, "n_factors": 100} | 0.9492 | 0.7315 | aef11a85c0784536bdf58f3bbfcd0093 |
| knn | {"k": 20, "sim_options": {"name": "cosine", "user_based": true}} | 1.0284 | 0.8099 | a2f1fbde4d5e41b7a030aa4a510d2755 |
| knn | {"k": 40, "sim_options": {"name": "cosine", "user_based": true}} | 1.0194 | 0.8038 | 2ea65f445b7c4f3f9c9d72bbf9d4f4db |
| knn | {"k": 40, "sim_options": {"name": "pearson", "user_based": true}} | 1.0150 | 0.8037 | ec700b290c40411c99ed66d91d3d1223 |

## Best Model

- Configuration: `{"lr_all": 0.005, "model_type": "svd", "n_epochs": 20, "n_factors": 50, "reg_all": 0.02}`
- RMSE: 0.9346
- MAE: 0.7376
- Run ID: `84f9d9a208dc4a72bc459078b47f7112`

## Recommendations

- Promote the `svd` configuration with the lowest RMSE (0.9346) for further validation.
- Review MLflow artifacts and prediction plots for the top runs before registration.
