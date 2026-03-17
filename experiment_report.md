# Experiment Report

Generated: 2026-03-17 21:13:33

## Summary

- Total experiments: 9
- Successful: 9
- Failed: 0

## Results

| Model | Parameters | RMSE | MAE | Run ID |
|-------|------------|------|-----|--------|
| svd | {"lr_all": 0.005, "n_epochs": 20, "n_factors": 50, "reg_all": 0.02} | 0.9352 | 0.7367 | b4aa3509ab9f433980f74772358568bd |
| svd | {"lr_all": 0.005, "n_epochs": 20, "n_factors": 100, "reg_all": 0.02} | 0.9327 | 0.7338 | 3df875caaf61407abb5b1a25491d47a4 |
| svd | {"lr_all": 0.01, "n_epochs": 30, "n_factors": 150, "reg_all": 0.02} | 0.9590 | 0.7541 | 8bf69a4f1f92433ea434de488a6a1aba |
| nmf | {"n_epochs": 30, "n_factors": 30} | 1.1715 | 0.8942 | 3c6a8f1dad774864bce372ac4795fbe8 |
| nmf | {"n_epochs": 50, "n_factors": 50} | 1.0333 | 0.7878 | 923940e906424a57b023534c5cbf4eba |
| nmf | {"n_epochs": 70, "n_factors": 100} | 0.9462 | 0.7305 | 13fb5c13f43a45b6a560210626962833 |
| knn | {"k": 20, "sim_options": {"name": "cosine", "user_based": true}} | 1.0284 | 0.8099 | 5f27ba2d8656432ab69cb3e5cd89c1ba |
| knn | {"k": 40, "sim_options": {"name": "cosine", "user_based": true}} | 1.0194 | 0.8038 | debbe245fb7748ed97e968d8e3c15653 |
| knn | {"k": 40, "sim_options": {"name": "pearson", "user_based": true}} | 1.0150 | 0.8037 | bc94c00a50ce471ca7967dee9b2a0422 |

## Best Model

- Configuration: `{"lr_all": 0.005, "model_type": "svd", "n_epochs": 20, "n_factors": 100, "reg_all": 0.02}`
- RMSE: 0.9327
- MAE: 0.7338
- Run ID: `3df875caaf61407abb5b1a25491d47a4`

## Recommendations

- Promote the `svd` configuration with the lowest RMSE (0.9327) for further validation.
- Review MLflow artifacts and prediction plots for the top runs before registration.
