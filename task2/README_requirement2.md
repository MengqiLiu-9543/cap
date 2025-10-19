# Requirement 2: Model Risk Factors and Time-to-Event Analysis

## AI-Powered Pharmacovigilance System for Oncology Drug Safety Monitoring

This implementation provides comprehensive survival analysis and risk factor modeling for oncology drug safety monitoring, focusing on time-to-event analysis using Cox proportional hazards models and identification of significant clinical and demographic predictors.

## Overview

Requirement 2 implements advanced survival analysis methods to:

1. **Model Risk Factors**: Identify significant clinical and demographic predictors of adverse events
2. **Time-to-Event Analysis**: Implement survival analysis models (Cox proportional hazards) to predict timing of serious adverse events
3. **Long-term Safety Outcomes**: Focus on modeling delayed safety outcomes like infections and secondary malignancies
4. **Feature Selection**: Use advanced techniques to identify the most important risk factors
5. **Comprehensive Validation**: Implement robust validation approaches and testing frameworks

## Key Features

### 🔬 **Survival Analysis Models**
- Cox Proportional Hazards Regression
- Kaplan-Meier Survival Curves
- Weibull Survival Models
- Time-to-event prediction

### 🎯 **Risk Factor Identification**
- Statistical feature selection (F-test, mutual information)
- Machine learning feature importance (Random Forest)
- Clinical risk factor validation
- Bootstrap stability analysis

### 🏥 **Long-term Adverse Events Focus**
- Infection analysis (neutropenic infections, opportunistic infections)
- Secondary malignancy detection
- Delayed safety outcome modeling
- Clinical pattern validation

### ✅ **Comprehensive Validation**
- Cross-validation with stratified sampling
- Bootstrap confidence intervals
- Model assumption testing
- Clinical relevance validation
- Benchmarking against standard methods

## Files Structure

```
requirement2/
├── requirement2_survival_analysis.py    # Main survival analysis implementation
├── requirement2_validation.py            # Validation and testing framework
├── requirement2_integration.py          # Complete integration script
├── requirements_requirement2.txt       # Python dependencies
└── README.md                            # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_requirement2.txt
```

### 2. Key Dependencies

- **lifelines**: Survival analysis library
- **scikit-survival**: Advanced survival analysis
- **scikit-learn**: Machine learning and feature selection
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **statsmodels**: Statistical modeling

## Usage

### Quick Start

Run the complete analysis with default settings:

```bash
python requirement2_integration.py
```

### Custom Analysis

```bash
# Analyze specific drugs
python requirement2_integration.py --drugs "Epcoritamab,Pembrolizumab,Nivolumab"

# Increase data collection limit
python requirement2_integration.py --limit 1000

# Specify output directory
python requirement2_integration.py --output-dir ./my_results/

# Enable verbose output
python requirement2_integration.py --verbose
```

### Validation Only

```bash
# Run validation on existing results
python requirement2_integration.py --validation-only --output-dir ./existing_results/
```

### Skip Data Collection

```bash
# Use existing data files
python requirement2_integration.py --skip-data-collection --output-dir ./existing_results/
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--drugs` | Comma-separated list of drugs to analyze | Key oncology drugs |
| `--limit` | Maximum records per drug | 500 |
| `--output-dir` | Output directory for results | `./requirement2_results/` |
| `--skip-data-collection` | Skip data collection, use existing files | False |
| `--validation-only` | Run only validation on existing results | False |
| `--verbose` | Enable verbose output | False |

## Output Structure

The analysis generates a comprehensive output structure:

```
requirement2_results/
├── data/
│   ├── combined_survival_data.csv           # Raw collected data
│   ├── survival_analysis_results.csv        # Processed data with predictions
│   └── survival_data_[DRUG].csv            # Individual drug data
├── models/
│   └── (model artifacts)
├── plots/
│   ├── kaplan_meier_curves.png             # Survival curves
│   ├── cox_hazard_ratios.png               # Risk factor importance
│   ├── long_term_analysis.png              # Long-term event analysis
│   ├── validation_summary.png              # Validation results
│   └── bootstrap_c_index.png               # Bootstrap confidence intervals
└── reports/
    ├── requirement2_final_report.md        # Comprehensive report
    ├── validation_report.txt               # Validation details
    └── validation_results.json             # Machine-readable validation
```

## Key Components

### 1. SurvivalDataCollector

Enhanced data collection specifically designed for survival analysis:

```python
from requirement2_survival_analysis import SurvivalDataCollector

collector = SurvivalDataCollector()
data = collector.collect_survival_data("Epcoritamab", limit=500)
```

**Features:**
- Time-to-event calculation using drug start/end dates
- Long-term adverse event classification
- Risk factor extraction
- Patient demographic processing

### 2. SurvivalAnalysisModel

Comprehensive survival analysis modeling:

```python
from requirement2_survival_analysis import SurvivalAnalysisModel

model = SurvivalAnalysisModel()
df_prepared = model.prepare_data(df)
features = model.perform_feature_selection(df_prepared)
cox_results = model.fit_cox_model(df_prepared, features)
```

**Features:**
- Cox proportional hazards modeling
- Kaplan-Meier survival curves
- Feature selection (statistical + ML methods)
- Long-term event analysis
- Risk prediction generation

### 3. SurvivalAnalysisValidator

Comprehensive validation framework:

```python
from requirement2_validation import run_comprehensive_validation

validation_results = run_comprehensive_validation(df, model_results)
```

**Features:**
- Cross-validation with stratified sampling
- Bootstrap confidence intervals
- Model assumption testing
- Clinical relevance validation
- Benchmarking against standard methods

## Methodology

### 1. Data Collection

- **Source**: FDA Adverse Event Reporting System (FAERS) via OpenFDA API
- **Focus**: Oncology drugs with emphasis on Epcoritamab case study
- **Fields**: Patient demographics, drug information, adverse events, timing data
- **Quality**: Data cleaning, deduplication, validation

### 2. Survival Analysis

- **Primary Model**: Cox Proportional Hazards Regression
- **Time Variable**: Days from drug start to adverse event
- **Event Variable**: Occurrence of serious adverse events
- **Covariates**: Age, sex, weight, polypharmacy, drug characteristics

### 3. Feature Selection

- **Statistical Methods**: F-test, mutual information
- **Machine Learning**: Random Forest feature importance
- **Clinical Validation**: Expert knowledge integration
- **Stability Analysis**: Bootstrap validation

### 4. Long-term Event Analysis

- **Infection Focus**: Neutropenic infections, opportunistic infections
- **Malignancy Focus**: Secondary malignancies, treatment-related cancers
- **Temporal Patterns**: Early vs. late event analysis
- **Risk Stratification**: Patient subgroup identification

### 5. Validation Framework

- **Cross-validation**: Stratified K-fold with survival-specific metrics
- **Bootstrap**: Confidence intervals for concordance index
- **Assumption Testing**: Proportional hazards, linearity, independence
- **Clinical Validation**: Pattern alignment with clinical expectations
- **Benchmarking**: Comparison with Random Forest Survival, Logistic Regression

## Key Metrics

### Model Performance

- **Concordance Index (C-index)**: Primary metric for survival models
- **Hazard Ratios**: Risk factor effect sizes
- **Confidence Intervals**: Statistical uncertainty quantification
- **Cross-validation**: Out-of-sample performance

### Clinical Relevance

- **Long-term Event Rates**: Infection and malignancy frequencies
- **Time-to-Event**: Median and mean times to adverse events
- **Risk Stratification**: High-risk patient identification
- **Pattern Validation**: Clinical expectation alignment

## Example Results

### Cox Model Performance
```
Concordance Index: 0.723
Cross-validation C-index: 0.715 ± 0.032
Bootstrap CI: [0.689, 0.741]
```

### Long-term Event Analysis
```
Overall Long-term Event Rate: 0.156
Infection Rate: 0.089
Secondary Malignancy Rate: 0.023
Median Time to Infection: 45.2 days
Median Time to Malignancy: 180.5 days
```

### Risk Factor Identification
```
Top Risk Factors (Hazard Ratios):
- Age Group (Elderly): HR = 1.847
- Polypharmacy: HR = 1.623
- Multiple Events: HR = 1.445
- Drug Interaction Risk: HR = 1.312
```

## Clinical Applications

### 1. Patient Risk Stratification
- Identify high-risk patients for enhanced monitoring
- Personalized safety surveillance protocols
- Targeted intervention strategies

### 2. Drug Safety Profiling
- Compare safety profiles across oncology drugs
- Support regulatory submissions
- Market positioning insights

### 3. Clinical Decision Support
- Risk-benefit assessment tools
- Treatment selection guidance
- Monitoring protocol optimization

## Advanced Features

### 1. Interactive Visualizations
- Kaplan-Meier survival curves by patient subgroups
- Risk factor importance plots
- Time-to-event distributions
- Clinical pattern validation charts

### 2. Export Capabilities
- Machine-readable results (JSON, CSV)
- Publication-ready figures (PNG, PDF)
- Comprehensive reports (Markdown)
- Validation documentation

### 3. Extensibility
- Modular design for easy extension
- Support for additional survival models
- Integration with other pharmacovigilance tools
- API-ready for web applications

## Troubleshooting

### Common Issues

1. **Data Collection Errors**
   - Check internet connection for API access
   - Verify drug name spelling
   - Reduce limit if timeout occurs

2. **Model Fitting Issues**
   - Ensure sufficient data (>50 records)
   - Check for missing values
   - Verify feature types

3. **Validation Errors**
   - Check data quality
   - Ensure proper data preparation
   - Verify model assumptions

### Performance Optimization

1. **Memory Usage**
   - Process drugs individually for large datasets
   - Use data chunking for very large collections
   - Monitor memory usage with verbose output

2. **Speed Optimization**
   - Reduce bootstrap iterations for faster validation
   - Use parallel processing for feature selection
   - Cache intermediate results

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Run tests: `python -m pytest tests/`
4. Follow coding standards

### Adding New Features

1. Extend existing classes
2. Add comprehensive tests
3. Update documentation
4. Validate with real data

## References

### Scientific Literature
- Cox, D.R. (1972). "Regression Models and Life-Tables". Journal of the Royal Statistical Society.
- Harrell, F.E. (2015). "Regression Modeling Strategies". Springer.
- Kleinbaum, D.G. (2012). "Survival Analysis: A Self-Learning Text". Springer.

### Technical Documentation
- [Lifelines Documentation](https://lifelines.readthedocs.io/)
- [Scikit-survival Documentation](https://scikit-survival.readthedocs.io/)
- [OpenFDA API Documentation](https://open.fda.gov/apis/)

### Clinical Guidelines
- FDA Guidance on Pharmacovigilance
- ICH E2E Guidelines on Pharmacovigilance Planning
- EMA Guidelines on Risk Management Plans

## License

This implementation is part of the AI-Powered Pharmacovigilance System developed for the NYU Center for Data Science Capstone Project in collaboration with Genmab.

## Contact

For questions or support regarding Requirement 2 implementation, please refer to the project documentation or contact the development team.

---

*This implementation provides a comprehensive solution for Requirement 2: Model Risk Factors and Time-to-Event Analysis, enabling advanced pharmacovigilance capabilities for oncology drug safety monitoring.*
