# Model Files Note

The trained model files (`.pkl` files) are excluded from this repository due to GitHub's file size limit (100MB).

## Excluded Files:
- `trained_model_random_forest.pkl` (105MB)
- `trained_model_gradient_boosting.pkl`
- `trained_model_logistic_regression.pkl`
- `trained_model_xgboost.pkl`
- `trained_model_calibrated.pkl`
- `crs_model_best.pkl`

## How to Generate Models:

To regenerate these model files, run the pipeline scripts:

```bash
# Train all models
python3 04_train_models.py

# Train CRS-specific model
python3 12_crs_model_training.py
```

The models will be saved locally and can be used for predictions and analysis.

