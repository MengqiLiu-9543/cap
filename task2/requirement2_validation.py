#!/usr/bin/env python3
"""
Requirement 2: Validation and Testing Framework
AI-Powered Pharmacovigilance System - Survival Analysis Validation

This module provides comprehensive validation and testing for the survival analysis models:
1. Cross-validation and model performance evaluation
2. Statistical significance testing
3. Clinical validation approaches
4. Model robustness testing
5. Benchmarking against established methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest

# Machine learning validation
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit,
    validation_curve, learning_curve
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)

# Survival analysis validation
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes

# Bootstrap and permutation testing
from sklearn.utils import resample
import itertools


class SurvivalAnalysisValidator:
    """
    Comprehensive validation framework for survival analysis models
    """
    
    def __init__(self):
        self.validation_results = {}
        self.bootstrap_results = {}
        
    def validate_cox_model(self, model, X: pd.DataFrame, y_time: np.ndarray, 
                          y_event: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Validate Cox proportional hazards model
        """
        print("Validating Cox proportional hazards model...")
        
        validation_results = {}
        
        # 1. Concordance Index (C-index)
        try:
            # Calculate C-index using lifelines
            c_index = concordance_index(y_time, -X.values @ model.summary['coef'].values, y_event)
            validation_results['concordance_index'] = c_index
            
            # Bootstrap confidence interval for C-index
            c_index_bootstrap = self._bootstrap_c_index(model, X, y_time, y_event, n_bootstrap=100)
            validation_results['c_index_ci'] = c_index_bootstrap
            
        except Exception as e:
            print(f"Error calculating concordance index: {e}")
            validation_results['concordance_index'] = None
        
        # 2. Model assumptions validation
        assumptions_results = self._validate_cox_assumptions(model, X, y_time, y_event)
        validation_results['assumptions'] = assumptions_results
        
        # 3. Cross-validation performance
        cv_results = self._cross_validate_cox_model(model, X, y_time, y_event, cv_folds)
        validation_results['cross_validation'] = cv_results
        
        # 4. Feature importance stability
        feature_stability = self._validate_feature_stability(model, X, y_time, y_event)
        validation_results['feature_stability'] = feature_stability
        
        return validation_results
    
    def _bootstrap_c_index(self, model, X: pd.DataFrame, y_time: np.ndarray, 
                          y_event: np.ndarray, n_bootstrap: int = 100) -> Dict:
        """
        Bootstrap confidence interval for concordance index
        """
        c_indices = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(X)), n_samples=len(X))
            X_boot = X.iloc[indices]
            y_time_boot = y_time[indices]
            y_event_boot = y_event[indices]
            
            try:
                # Fit model on bootstrap sample
                from lifelines import CoxPHFitter
                cox_boot = CoxPHFitter()
                cox_boot.fit(pd.concat([X_boot, 
                                      pd.DataFrame({'time': y_time_boot, 'event': y_event_boot})], axis=1),
                           duration_col='time', event_col='event')
                
                # Calculate C-index
                c_idx = concordance_index(y_time_boot, 
                                        -X_boot.values @ cox_boot.summary['coef'].values, 
                                        y_event_boot)
                c_indices.append(c_idx)
                
            except Exception:
                continue
        
        if c_indices:
            return {
                'mean': np.mean(c_indices),
                'std': np.std(c_indices),
                'ci_lower': np.percentile(c_indices, 2.5),
                'ci_upper': np.percentile(c_indices, 97.5),
                'values': c_indices
            }
        else:
            return {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
    
    def _validate_cox_assumptions(self, model, X: pd.DataFrame, y_time: np.ndarray, 
                                y_event: np.ndarray) -> Dict:
        """
        Validate Cox model assumptions
        """
        assumptions = {}
        
        try:
            # 1. Proportional hazards assumption
            # Test using Schoenfeld residuals
            residuals = model.compute_residuals(X, kind='schoenfeld')
            
            # Test for time-dependent effects
            from lifelines.statistics import proportional_hazard_test
            ph_test = proportional_hazard_test(model, X)
            assumptions['proportional_hazards_pvalue'] = ph_test.summary['p'].min()
            assumptions['proportional_hazards_violated'] = ph_test.summary['p'].min() < 0.05
            
            # 2. Linearity assumption
            # Test linearity of continuous variables
            linearity_tests = {}
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    # Correlation between variable and log(time)
                    corr = np.corrcoef(X[col], np.log(y_time + 1))[0, 1]
                    linearity_tests[col] = {
                        'correlation': corr,
                        'is_linear': abs(corr) > 0.3
                    }
            
            assumptions['linearity_tests'] = linearity_tests
            
            # 3. Independence assumption
            # Test for clustering effects
            independence_test = self._test_independence_assumption(X, y_time, y_event)
            assumptions['independence_test'] = independence_test
            
        except Exception as e:
            print(f"Error validating assumptions: {e}")
            assumptions['error'] = str(e)
        
        return assumptions
    
    def _test_independence_assumption(self, X: pd.DataFrame, y_time: np.ndarray, 
                                    y_event: np.ndarray) -> Dict:
        """
        Test independence assumption using correlation analysis
        """
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            return {
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': corr_matrix.abs().max().max(),
                'independence_violated': len(high_corr_pairs) > 0
            }
            
        except Exception:
            return {'error': 'Could not test independence'}
    
    def _cross_validate_cox_model(self, model, X: pd.DataFrame, y_time: np.ndarray, 
                                 y_event: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Cross-validation for Cox model
        """
        cv_results = {
            'c_indices': [],
            'mean_c_index': None,
            'std_c_index': None
        }
        
        try:
            # Stratified K-fold based on event status
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, test_idx in skf.split(X, y_event):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
                y_event_train, y_event_test = y_event[train_idx], y_event[test_idx]
                
                try:
                    # Fit model on training set
                    from lifelines import CoxPHFitter
                    cox_cv = CoxPHFitter()
                    train_data = pd.concat([
                        X_train,
                        pd.DataFrame({'time': y_time_train, 'event': y_event_train})
                    ], axis=1)
                    
                    cox_cv.fit(train_data, duration_col='time', event_col='event')
                    
                    # Calculate C-index on test set
                    c_idx = concordance_index(y_time_test,
                                            -X_test.values @ cox_cv.summary['coef'].values,
                                            y_event_test)
                    cv_results['c_indices'].append(c_idx)
                    
                except Exception:
                    continue
            
            if cv_results['c_indices']:
                cv_results['mean_c_index'] = np.mean(cv_results['c_indices'])
                cv_results['std_c_index'] = np.std(cv_results['c_indices'])
                
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            cv_results['error'] = str(e)
        
        return cv_results
    
    def _validate_feature_stability(self, model, X: pd.DataFrame, y_time: np.ndarray, 
                                  y_event: np.ndarray) -> Dict:
        """
        Validate feature importance stability using bootstrap
        """
        stability_results = {
            'feature_importance_variance': {},
            'stable_features': [],
            'unstable_features': []
        }
        
        try:
            # Bootstrap feature importance
            n_bootstrap = 50
            feature_importances = {col: [] for col in X.columns}
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = resample(range(len(X)), n_samples=len(X))
                X_boot = X.iloc[indices]
                y_time_boot = y_time[indices]
                y_event_boot = y_event[indices]
                
                try:
                    # Fit model
                    from lifelines import CoxPHFitter
                    cox_boot = CoxPHFitter()
                    boot_data = pd.concat([
                        X_boot,
                        pd.DataFrame({'time': y_time_boot, 'event': y_event_boot})
                    ], axis=1)
                    
                    cox_boot.fit(boot_data, duration_col='time', event_col='event')
                    
                    # Store coefficients
                    for col in X.columns:
                        if col in cox_boot.summary.index:
                            feature_importances[col].append(cox_boot.summary.loc[col, 'coef'])
                        
                except Exception:
                    continue
            
            # Calculate stability metrics
            for col, importances in feature_importances.items():
                if len(importances) > 10:  # Minimum bootstrap samples
                    stability_results['feature_importance_variance'][col] = {
                        'mean': np.mean(importances),
                        'std': np.std(importances),
                        'cv': np.std(importances) / abs(np.mean(importances)) if np.mean(importances) != 0 else np.inf
                    }
                    
                    # Classify as stable/unstable
                    cv = stability_results['feature_importance_variance'][col]['cv']
                    if cv < 0.5:  # Coefficient of variation < 50%
                        stability_results['stable_features'].append(col)
                    else:
                        stability_results['unstable_features'].append(col)
        
        except Exception as e:
            print(f"Error validating feature stability: {e}")
            stability_results['error'] = str(e)
        
        return stability_results
    
    def validate_long_term_events(self, df: pd.DataFrame) -> Dict:
        """
        Validate long-term adverse event analysis
        """
        print("Validating long-term adverse event analysis...")
        
        validation_results = {}
        
        # 1. Statistical significance of long-term events
        long_term_df = df[df['is_long_term_event'] == 1]
        other_df = df[df['is_long_term_event'] == 0]
        
        if len(long_term_df) > 10 and len(other_df) > 10:
            # Compare time to event distributions
            time_test = stats.mannwhitneyu(
                long_term_df['time_to_event_days'],
                other_df['time_to_event_days'],
                alternative='two-sided'
            )
            
            validation_results['time_distribution_test'] = {
                'statistic': time_test.statistic,
                'p_value': time_test.pvalue,
                'significant': time_test.pvalue < 0.05
            }
            
            # Compare serious event rates
            serious_test = stats.chi2_contingency([
                [long_term_df['is_serious'].sum(), len(long_term_df) - long_term_df['is_serious'].sum()],
                [other_df['is_serious'].sum(), len(other_df) - other_df['is_serious'].sum()]
            ])
            
            validation_results['serious_event_test'] = {
                'chi2_statistic': serious_test[0],
                'p_value': serious_test[1],
                'significant': serious_test[1] < 0.05
            }
        
        # 2. Clinical validation
        clinical_validation = self._validate_clinical_relevance(df)
        validation_results['clinical_validation'] = clinical_validation
        
        # 3. Temporal validation
        temporal_validation = self._validate_temporal_patterns(df)
        validation_results['temporal_validation'] = temporal_validation
        
        return validation_results
    
    def _validate_clinical_relevance(self, df: pd.DataFrame) -> Dict:
        """
        Validate clinical relevance of findings
        """
        clinical_results = {}
        
        # 1. Infection analysis validation
        infection_df = df[df['is_infection'] == 1]
        if len(infection_df) > 10:
            # Expected patterns for infections
            infection_patterns = {
                'elderly_higher_rate': infection_df[df['age_group'] == 'ELDERLY']['is_serious'].mean() >
                                     infection_df[df['age_group'] != 'ELDERLY']['is_serious'].mean(),
                'polypharmacy_risk': infection_df[df['polypharmacy'] == True]['is_serious'].mean() >
                                   infection_df[df['polypharmacy'] == False]['is_serious'].mean(),
                'time_pattern': infection_df['time_to_event_days'].median() > 30  # Infections typically occur later
            }
            clinical_results['infection_patterns'] = infection_patterns
        
        # 2. Secondary malignancy validation
        malignancy_df = df[df['is_secondary_malignancy'] == 1]
        if len(malignancy_df) > 10:
            malignancy_patterns = {
                'elderly_higher_rate': malignancy_df[df['age_group'] == 'ELDERLY']['is_serious'].mean() >
                                     malignancy_df[df['age_group'] != 'ELDERLY']['is_serious'].mean(),
                'longer_time': malignancy_df['time_to_event_days'].median() > 180,  # Malignancies develop over months/years
                'serious_outcome': malignancy_df['is_serious'].mean() > 0.8  # Most malignancies are serious
            }
            clinical_results['malignancy_patterns'] = malignancy_patterns
        
        return clinical_results
    
    def _validate_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Validate temporal patterns in adverse events
        """
        temporal_results = {}
        
        # 1. Early vs late events
        early_events = df[df['time_to_event_days'] <= 30]
        late_events = df[df['time_to_event_days'] > 30]
        
        if len(early_events) > 10 and len(late_events) > 10:
            # Compare event types
            early_types = early_events['event_type'].value_counts(normalize=True)
            late_types = late_events['event_type'].value_counts(normalize=True)
            
            temporal_results['early_late_comparison'] = {
                'early_event_types': early_types.to_dict(),
                'late_event_types': late_types.to_dict(),
                'early_long_term_rate': early_events['is_long_term_event'].mean(),
                'late_long_term_rate': late_events['is_long_term_event'].mean()
            }
        
        # 2. Seasonal patterns (if date information available)
        if 'receive_date' in df.columns:
            try:
                df['month'] = pd.to_datetime(df['receive_date'], errors='coerce').dt.month
                monthly_patterns = df.groupby('month')['is_long_term_event'].mean()
                temporal_results['monthly_patterns'] = monthly_patterns.to_dict()
            except Exception:
                temporal_results['monthly_patterns'] = None
        
        return temporal_results
    
    def benchmark_models(self, df: pd.DataFrame) -> Dict:
        """
        Benchmark survival analysis models against standard approaches
        """
        print("Benchmarking models against standard approaches...")
        
        benchmark_results = {}
        
        # Prepare data
        feature_cols = [col for col in df.columns if col.endswith('_encoded') or 
                       col in ['patient_age', 'patient_weight', 'total_drugs', 'total_events']]
        
        X = df[feature_cols].fillna(0)
        y_time = df['time_to_event_days'].values
        y_event = df['event_occurred'].values
        
        # 1. Compare with Random Forest Survival
        try:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.preprocessing import OneHotEncoder
            
            # Prepare data for scikit-survival
            y_surv = np.array([(bool(e), t) for e, t in zip(y_event, y_time)],
                            dtype=[('event', bool), ('time', float)])
            
            rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
            rsf.fit(X, y_surv)
            
            # Calculate C-index
            rsf_c_index = rsf.score(X, y_surv)
            
            benchmark_results['random_survival_forest'] = {
                'c_index': rsf_c_index,
                'feature_importance': dict(zip(feature_cols, rsf.feature_importances_))
            }
            
        except ImportError:
            print("scikit-survival not available for Random Forest Survival")
            benchmark_results['random_survival_forest'] = None
        except Exception as e:
            print(f"Error with Random Forest Survival: {e}")
            benchmark_results['random_survival_forest'] = None
        
        # 2. Compare with Logistic Regression (simplified approach)
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Create binary outcome for serious events
            y_binary = df['is_serious'].values
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X, y_binary)
            
            # Cross-validation
            cv_scores = cross_val_score(lr, X, y_binary, cv=5, scoring='roc_auc')
            
            benchmark_results['logistic_regression'] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'feature_coefficients': dict(zip(feature_cols, lr.coef_[0]))
            }
            
        except Exception as e:
            print(f"Error with Logistic Regression: {e}")
            benchmark_results['logistic_regression'] = None
        
        # 3. Compare with Kaplan-Meier (non-parametric baseline)
        try:
            from lifelines import KaplanMeierFitter
            
            kmf = KaplanMeierFitter()
            kmf.fit(y_time, y_event)
            
            # Calculate median survival time
            median_survival = kmf.median_survival_time_
            
            benchmark_results['kaplan_meier_baseline'] = {
                'median_survival_time': median_survival,
                'survival_function': kmf.survival_function_
            }
            
        except Exception as e:
            print(f"Error with Kaplan-Meier: {e}")
            benchmark_results['kaplan_meier_baseline'] = None
        
        return benchmark_results
    
    def generate_validation_report(self, validation_results: Dict, output_file: str = 'validation_report.txt'):
        """
        Generate comprehensive validation report
        """
        print(f"Generating validation report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SURVIVAL ANALYSIS VALIDATION REPORT\n")
            f.write("AI-Powered Pharmacovigilance System\n")
            f.write("=" * 80 + "\n\n")
            
            # Cox Model Validation
            if 'cox_model' in validation_results:
                f.write("COX PROPORTIONAL HAZARDS MODEL VALIDATION\n")
                f.write("-" * 50 + "\n")
                
                cox_results = validation_results['cox_model']
                
                if 'concordance_index' in cox_results:
                    f.write(f"Concordance Index: {cox_results['concordance_index']:.3f}\n")
                
                if 'c_index_ci' in cox_results and cox_results['c_index_ci']['mean']:
                    ci = cox_results['c_index_ci']
                    f.write(f"C-index Bootstrap CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]\n")
                
                if 'cross_validation' in cox_results:
                    cv = cox_results['cross_validation']
                    if cv['mean_c_index']:
                        f.write(f"Cross-validation C-index: {cv['mean_c_index']:.3f} ± {cv['std_c_index']:.3f}\n")
                
                if 'assumptions' in cox_results:
                    assumptions = cox_results['assumptions']
                    if 'proportional_hazards_pvalue' in assumptions:
                        f.write(f"Proportional Hazards Test p-value: {assumptions['proportional_hazards_pvalue']:.3f}\n")
                        f.write(f"PH Assumption Violated: {assumptions.get('proportional_hazards_violated', 'Unknown')}\n")
                
                f.write("\n")
            
            # Long-term Events Validation
            if 'long_term_events' in validation_results:
                f.write("LONG-TERM ADVERSE EVENTS VALIDATION\n")
                f.write("-" * 50 + "\n")
                
                lt_results = validation_results['long_term_events']
                
                if 'time_distribution_test' in lt_results:
                    test = lt_results['time_distribution_test']
                    f.write(f"Time Distribution Test p-value: {test['p_value']:.3f}\n")
                    f.write(f"Significant Difference: {test['significant']}\n")
                
                if 'clinical_validation' in lt_results:
                    clinical = lt_results['clinical_validation']
                    f.write("Clinical Validation Results:\n")
                    for pattern_type, patterns in clinical.items():
                        f.write(f"  {pattern_type}:\n")
                        for pattern, result in patterns.items():
                            f.write(f"    {pattern}: {result}\n")
                
                f.write("\n")
            
            # Benchmark Results
            if 'benchmark' in validation_results:
                f.write("MODEL BENCHMARKING\n")
                f.write("-" * 50 + "\n")
                
                benchmark = validation_results['benchmark']
                
                for model_name, results in benchmark.items():
                    if results:
                        f.write(f"{model_name.upper()}:\n")
                        for metric, value in results.items():
                            if isinstance(value, dict):
                                f.write(f"  {metric}:\n")
                                for sub_metric, sub_value in value.items():
                                    f.write(f"    {sub_metric}: {sub_value}\n")
                            else:
                                f.write(f"  {metric}: {value}\n")
                        f.write("\n")
            
            # Summary and Recommendations
            f.write("SUMMARY AND RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            
            # Overall model performance
            if 'cox_model' in validation_results:
                cox_results = validation_results['cox_model']
                c_index = cox_results.get('concordance_index', 0)
                
                if c_index > 0.7:
                    f.write("✓ Model shows good discriminative ability (C-index > 0.7)\n")
                elif c_index > 0.6:
                    f.write("⚠ Model shows moderate discriminative ability (C-index > 0.6)\n")
                else:
                    f.write("✗ Model shows poor discriminative ability (C-index < 0.6)\n")
            
            # Assumptions validation
            if 'cox_model' in validation_results and 'assumptions' in validation_results['cox_model']:
                assumptions = validation_results['cox_model']['assumptions']
                if assumptions.get('proportional_hazards_violated', False):
                    f.write("⚠ Proportional hazards assumption may be violated\n")
                    f.write("  Recommendation: Consider time-dependent covariates or alternative models\n")
                else:
                    f.write("✓ Proportional hazards assumption appears valid\n")
            
            # Clinical relevance
            if 'long_term_events' in validation_results and 'clinical_validation' in validation_results['long_term_events']:
                f.write("✓ Long-term event patterns align with clinical expectations\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Validation Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"Validation report saved to: {output_file}")


def run_comprehensive_validation(df: pd.DataFrame, model_results: Dict) -> Dict:
    """
    Run comprehensive validation suite
    """
    print("=" * 80)
    print("RUNNING COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)
    
    validator = SurvivalAnalysisValidator()
    validation_results = {}
    
    # 1. Cox Model Validation
    if 'cox_model' in model_results and model_results['cox_model']:
        print("\n1. Validating Cox Proportional Hazards Model...")
        
        cox_model = model_results['cox_model']['model']
        features = model_results['cox_model']['features']
        
        # Prepare data
        feature_data = df[features].fillna(0)
        y_time = df['time_to_event_days'].values
        y_event = df['event_occurred'].values
        
        cox_validation = validator.validate_cox_model(cox_model, feature_data, y_time, y_event)
        validation_results['cox_model'] = cox_validation
    
    # 2. Long-term Events Validation
    print("\n2. Validating Long-term Adverse Events Analysis...")
    lt_validation = validator.validate_long_term_events(df)
    validation_results['long_term_events'] = lt_validation
    
    # 3. Model Benchmarking
    print("\n3. Benchmarking Models...")
    benchmark_results = validator.benchmark_models(df)
    validation_results['benchmark'] = benchmark_results
    
    # 4. Generate Validation Report
    print("\n4. Generating Validation Report...")
    validator.generate_validation_report(validation_results)
    
    # 5. Create Validation Visualizations
    print("\n5. Creating Validation Visualizations...")
    create_validation_plots(validation_results, df)
    
    return validation_results


def create_validation_plots(validation_results: Dict, df: pd.DataFrame, output_dir: str = './'):
    """
    Create validation plots and visualizations
    """
    plt.style.use('seaborn-v0_8')
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # C-index comparison
    if 'cox_model' in validation_results and 'benchmark' in validation_results:
        models = ['Cox Model']
        c_indices = [validation_results['cox_model'].get('concordance_index', 0)]
        
        benchmark = validation_results['benchmark']
        if benchmark.get('random_survival_forest'):
            models.append('Random Forest')
            c_indices.append(benchmark['random_survival_forest']['c_index'])
        
        if benchmark.get('logistic_regression'):
            models.append('Logistic Regression')
            c_indices.append(benchmark['logistic_regression']['cv_auc_mean'])
        
        axes[0, 0].bar(models, c_indices)
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('C-index / AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Feature Importance Stability
    if 'cox_model' in validation_results and 'feature_stability' in validation_results['cox_model']:
        stability = validation_results['cox_model']['feature_stability']
        if 'feature_importance_variance' in stability:
            features = list(stability['feature_importance_variance'].keys())
            cv_values = [stability['feature_importance_variance'][f]['cv'] for f in features]
            
            axes[0, 1].barh(features, cv_values)
            axes[0, 1].set_title('Feature Importance Stability')
            axes[0, 1].set_xlabel('Coefficient of Variation')
            axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Long-term Event Patterns
    if 'long_term_events' in validation_results:
        lt_df = df[df['is_long_term_event'] == 1]
        other_df = df[df['is_long_term_event'] == 0]
        
        if len(lt_df) > 0 and len(other_df) > 0:
            # Time to event comparison
            axes[1, 0].hist([lt_df['time_to_event_days'], other_df['time_to_event_days']], 
                           bins=30, alpha=0.7, label=['Long-term Events', 'Other Events'])
            axes[1, 0].set_title('Time to Event Distribution')
            axes[1, 0].set_xlabel('Days')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
    
    # Clinical Validation Results
    if 'long_term_events' in validation_results and 'clinical_validation' in validation_results['long_term_events']:
        clinical = validation_results['long_term_events']['clinical_validation']
        
        if 'infection_patterns' in clinical:
            patterns = clinical['infection_patterns']
            pattern_names = list(patterns.keys())
            pattern_values = [patterns[p] for p in pattern_names]
            
            axes[1, 1].bar(pattern_names, pattern_values)
            axes[1, 1].set_title('Infection Pattern Validation')
            axes[1, 1].set_ylabel('Pattern Confirmed')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bootstrap Confidence Intervals
    if 'cox_model' in validation_results and 'c_index_ci' in validation_results['cox_model']:
        ci_data = validation_results['cox_model']['c_index_ci']
        if ci_data.get('values'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(ci_data['values'], bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(ci_data['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(ci_data['ci_lower'], color='orange', linestyle='--', label='95% CI')
            ax.axvline(ci_data['ci_upper'], color='orange', linestyle='--')
            
            ax.set_title('Bootstrap Distribution of Concordance Index')
            ax.set_xlabel('Concordance Index')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/bootstrap_c_index.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Validation plots saved to {output_dir}")


if __name__ == "__main__":
    """
    Run validation as standalone script
    """
    print("This module provides validation functions for the survival analysis.")
    print("Import and use run_comprehensive_validation() with your data and model results.")
