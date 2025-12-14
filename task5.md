# Task 5: Drug Adverse Event Severity Prediction Pipeline

## Overview

This task implements a comprehensive machine learning pipeline for predicting adverse event (AE) severity, with a specific focus on Cytokine Release Syndrome (CRS) mortality prediction for patients treated with Epcoritamab.

## Project Structure

```
task5_severity_prediction/
├── 01_extract_data.py              # Data extraction from FAERS API
├── 02_inspect_data.py              # Data quality inspection
├── 03_preprocess_data.py           # Data preprocessing and feature engineering
├── 04_train_models.py              # General severity prediction model training
├── 05_analyze_features.py          # Feature importance analysis
├── 06_visualize_results.py        # Visualization of results
├── 07_explainability.py            # Model explainability analysis
├── 08_test_models.py               # Model testing and evaluation
├── 09_test_epcoritamab.py          # Epcoritamab-specific testing
├── 11_granular_crs_analysis.py     # Granular CRS risk stratification
├── 12_crs_model_training.py        # CRS mortality prediction model
├── 13_crs_shap_analysis.py        # SHAP analysis for CRS model
├── generate_presentation_charts.py # Generate presentation charts
├── pipeline.py                     # Main pipeline orchestrator
└── pipeline_full_run.py            # Full pipeline runner with logging
```

## Key Features

### 1. Data Extraction (01_extract_data.py)
- Extracts drug adverse event data from FDA FAERS API
- Supports parameterized drug and adverse event queries
- Handles API rate limiting and retries
- Deduplicates records based on case ID

### 2. Data Preprocessing (03_preprocess_data.py)
- Feature engineering:
  - Age stratification
  - BMI calculation and categorization
  - Comorbidity flags
  - Drug category classification
  - Cancer stage extraction from free text
  - Polypharmacy analysis
- Missing value imputation
- Data normalization

### 3. Model Training (04_train_models.py, 12_crs_model_training.py)
- Multiple ML algorithms:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - XGBoost (if available)
- Model evaluation metrics:
  - PR-AUC, ROC-AUC
  - F1-score, Precision, Recall
  - Confusion Matrix
- Best model selection based on PR-AUC

### 4. Granular Analysis (11_granular_crs_analysis.py)
- Stratified risk analysis by:
  - Age groups
  - Cancer stage
  - BMI categories
  - Polypharmacy levels
  - Drug combinations
  - Comorbidities
- Generates detailed risk stratification reports

### 5. Model Explainability (07_explainability.py, 13_crs_shap_analysis.py)
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Local and global interpretability
- Plain language summaries

### 6. Presentation Charts (generate_presentation_charts.py)
- SHAP summary plots for model interpretation
- Granular risk stratification bar charts
- Publication-ready visualizations

## Usage

### Quick Start

```python
from pipeline import run_pipeline

# Run complete pipeline for Epcoritamab and CRS
results = run_pipeline(
    drug="Epcoritamab",
    adverse_event="CRS",
    max_records=500
)
```

### Step-by-Step Execution

```bash
# Step 1: Extract data
python 01_extract_data.py

# Step 2: Inspect data quality
python 02_inspect_data.py

# Step 3: Preprocess data
python 03_preprocess_data.py

# Step 4: Train general models
python 04_train_models.py

# Step 5: Analyze features
python 05_analyze_features.py

# Step 6: Visualize results
python 06_visualize_results.py

# Step 7: Explainability
python 07_explainability.py

# Step 8: Test models
python 08_test_models.py

# Step 9: Test Epcoritamab specifically
python 09_test_epcoritamab.py

# Step 11: Granular CRS analysis
python 11_granular_crs_analysis.py

# Step 12: Train CRS mortality model
python 12_crs_model_training.py

# Step 13: SHAP analysis for CRS model
python 13_crs_shap_analysis.py

# Generate presentation charts
python generate_presentation_charts.py
```

### Full Pipeline Run

```bash
# Run all steps with logging
python pipeline_full_run.py

# Results will be saved to pipeline_full_run.txt
```

## Key Results

### CRS Mortality Model Performance
- **Dataset**: 185 CRS patients, 150 deaths (81.1% mortality)
- **Best Model**: Random Forest
- **PR-AUC**: 0.896
- **F1-Score**: 0.767
- **Accuracy**: 0.643
- **Precision**: 0.805
- **Recall**: 0.733

### Top Risk Factors (from SHAP analysis)
1. Number of concurrent drugs (polypharmacy)
2. Age
3. Patient weight
4. Antiviral medication use
5. Chemotherapy use
6. Number of adverse reactions
7. BMI
8. Targeted therapy use
9. Patient onset age
10. Patient sex

### Granular Risk Stratification
- **Age 75+**: Highest death rate
- **Stage IV Cancer**: Higher risk than earlier stages
- **High Polypharmacy (>10 drugs)**: Significantly increased risk
- **Obesity (BMI>30)**: Elevated risk

## Dependencies

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
shap (for explainability)
xgboost (optional, for XGBoost model)
requests (for API calls)
```

## Data Sources

- **FAERS (FDA Adverse Event Reporting System)**: Primary data source
- **Fields Used**:
  - Patient demographics (age, sex, weight)
  - Drug information (concurrent medications)
  - Adverse events (reactions)
  - Outcomes (death, hospitalization, etc.)
  - Indications (cancer stage from free text)

## Methodology

1. **Data Extraction**: Query FAERS API for specific drug-AE combinations
2. **Feature Engineering**: Create predictive features from raw data
3. **Model Training**: Train multiple ML models and select best performer
4. **Evaluation**: Comprehensive metrics on test set
5. **Explainability**: SHAP analysis to understand model decisions
6. **Stratification**: Granular analysis by patient subgroups

## Limitations

1. **Cancer Stage**: Not a structured field in FAERS; extracted imperfectly from free text
2. **Missing Data**: Some features have high missing rates; imputed with medians
3. **Data Quality**: FAERS is a spontaneous reporting system with inherent biases
4. **Sample Size**: CRS dataset (n=185) is relatively small for deep learning approaches

## Future Directions

1. **Biomarker Integration**: Framework ready for IL-6, CRP, Ferritin data
2. **Structured Cancer Stage**: When available, will improve model performance
3. **Deep Learning**: Larger datasets could benefit from neural networks
4. **Real-time Prediction**: Deploy model for clinical decision support

## Output Files

- `main_data.csv`: Extracted raw data
- `preprocessed_data.csv`: Preprocessed data with engineered features
- `trained_model_*.pkl`: Trained model files
- `crs_model_best.pkl`: Best CRS mortality model
- `crs_model_meta.json`: Model metadata
- `crs_shap_summary_presentation.png`: SHAP summary plot
- `crs_granular_risk_stratification.png`: Risk stratification chart
- `pipeline_full_run.txt`: Complete execution log

## Contact

For questions or issues, please refer to the main project repository.

## License

See main project license.

