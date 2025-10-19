#!/usr/bin/env python3
"""
Requirement 2: Model Risk Factors and Time-to-Event Analysis
AI-Powered Pharmacovigilance System for Oncology Drug Safety Monitoring

This module implements:
1. Cox proportional hazards models for survival analysis
2. Feature selection techniques to identify significant predictors
3. Focus on long-term adverse events (infections, secondary malignancies)
4. Comprehensive risk factor modeling system
5. Validation approaches and testing framework

Key Features:
- Time-to-event analysis using drug start/end dates and adverse event timing
- Risk factor identification for serious adverse events
- Long-term safety outcome modeling
- Patient demographic and clinical predictor analysis
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Survival Analysis Libraries
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Statistical Libraries
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

# Enhanced data collection for survival analysis
class SurvivalDataCollector:
    """
    Enhanced data collector specifically designed for survival analysis
    Focuses on time-to-event data and risk factors
    """
    
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/event.json"
        # 35种常见肿瘤药物（聚焦靶向治疗和免疫治疗）
        self.oncology_drugs = [
            "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",
            "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab", "Epcoritamab",
            "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
            "Paclitaxel", "Docetaxel", "Doxorubicin", "Carboplatin", "Cisplatin",
            "Lenalidomide", "Pomalidomide", "Bortezomib", "Carfilzomib", "Venetoclax",
            "Ibrutinib", "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
            "Palbociclib", "Ribociclib", "Abemaciclib", "Vemurafenib", "Dabrafenib"
        ]
        
        # Long-term adverse events of interest
        self.long_term_events = [
            "INFECTION", "SECONDARY MALIGNANCY", "CARDIOVASCULAR DISORDER",
            "NEUROLOGICAL DISORDER", "RENAL DISORDER", "HEPATIC DISORDER",
            "PULMONARY DISORDER", "ENDOCRINE DISORDER", "IMMUNE DISORDER",
            "METABOLIC DISORDER", "SKELETAL DISORDER", "DERMATOLOGICAL DISORDER"
        ]
        
        # Infection-related terms
        self.infection_terms = [
            "INFECTION", "SEPSIS", "PNEUMONIA", "BACTERIAL INFECTION",
            "FUNGAL INFECTION", "VIRAL INFECTION", "OPPORTUNISTIC INFECTION",
            "NEUTROPENIC INFECTION", "FEBRILE NEUTROPENIA"
        ]
        
        # Secondary malignancy terms
        self.malignancy_terms = [
            "SECONDARY MALIGNANCY", "SECONDARY NEOPLASM", "METASTASIS",
            "CARCINOMA", "SARCOMA", "LYMPHOMA", "LEUKEMIA", "MYELOMA"
        ]

    def collect_survival_data(self, drug_name: str, limit: int = 1000) -> List[Dict]:
        """
        Collect comprehensive data for survival analysis
        Includes time-to-event information and risk factors
        """
        print(f"Collecting survival analysis data for {drug_name}...")
        
        all_records = []
        skip = 0
        
        while len(all_records) < limit:
            try:
                params = {
                    'search': f'patient.drug.openfda.generic_name:"{drug_name}"',
                    'limit': min(100, limit - len(all_records)),
                    'skip': skip
                }
                
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        break
                    
                    for record in results:
                        processed = self.process_survival_record(record, drug_name)
                        if processed:
                            all_records.extend(processed)
                    
                    skip += len(results)
                    time.sleep(0.3)
                    
                else:
                    break
                    
            except Exception as e:
                print(f"Error collecting data for {drug_name}: {str(e)}")
                break
        
        print(f"Collected {len(all_records)} records for {drug_name}")
        return all_records

    def process_survival_record(self, record: Dict, target_drug: str) -> List[Dict]:
        """
        Process individual record for survival analysis
        Extract time-to-event data and risk factors
        """
        try:
            # Basic record information
            safety_id = record.get('safetyreportid', '')
            receive_date = record.get('receivedate', '')
            
            # Patient demographics
            patient = record.get('patient', {})
            age = patient.get('patientonsetage', '')
            age_unit = patient.get('patientonsetageunit', '')
            sex = patient.get('patientsex', '')
            weight = patient.get('patientweight', '')
            
            # Death information
            death_info = patient.get('patientdeath', {})
            death_date = death_info.get('patientdeathdate', '')
            
            # Seriousness indicators
            serious = record.get('serious', 0)
            seriousness_death = record.get('seriousnessdeath', 0)
            seriousness_hosp = record.get('seriousnesshospitalization', 0)
            seriousness_life = record.get('seriousnesslifethreatening', 0)
            seriousness_disable = record.get('seriousnessdisabling', 0)
            
            # Drug information with timing
            drugs = patient.get('drug', [])
            drug_data = []
            target_drug_data = None
            
            for drug in drugs:
                openfda = drug.get('openfda', {})
                generic_names = openfda.get('generic_name', [])
                
                drug_info = {
                    'generic_name': '|'.join(generic_names),
                    'start_date': drug.get('drugstartdate', ''),
                    'end_date': drug.get('drugenddate', ''),
                    'treatment_duration': drug.get('drugtreatmentduration', ''),
                    'treatment_duration_unit': drug.get('drugtreatmentdurationunit', ''),
                    'indication': drug.get('drugindication', ''),
                    'dosage_form': drug.get('drugdosageform', ''),
                    'administration_route': drug.get('drugadministrationroute', ''),
                    'characterization': drug.get('drugcharacterization', ''),
                    'action_drug': drug.get('actiondrug', '')
                }
                
                drug_data.append(drug_info)
                
                # Check if this is our target drug
                if target_drug.lower() in [name.lower() for name in generic_names]:
                    target_drug_data = drug_info
            
            # Reaction information
            reactions = patient.get('reaction', [])
            adverse_events = []
            reaction_outcomes = []
            
            for reaction in reactions:
                ae_term = reaction.get('reactionmeddrapt', '')
                outcome = reaction.get('reactionoutcome', '')
                if ae_term:
                    adverse_events.append(ae_term)
                    reaction_outcomes.append(outcome)
            
            # Create survival analysis records
            survival_records = []
            
            for i, ae in enumerate(adverse_events):
                # Calculate time-to-event
                time_to_event = self.calculate_time_to_event(
                    target_drug_data, receive_date, death_date, serious
                )
                
                # Determine event type
                event_type = self.classify_adverse_event(ae)
                
                # Risk factors
                risk_factors = self.extract_risk_factors(
                    age, age_unit, sex, weight, drug_data, adverse_events
                )
                
                survival_record = {
                    'safety_report_id': safety_id,
                    'receive_date': receive_date,
                    'target_drug': target_drug,
                    'adverse_event': ae,
                    'event_type': event_type,
                    'is_long_term_event': event_type in ['INFECTION', 'SECONDARY_MALIGNANCY'],
                    'is_infection': event_type == 'INFECTION',
                    'is_secondary_malignancy': event_type == 'SECONDARY_MALIGNANCY',
                    
                    # Time-to-event data
                    'time_to_event_days': time_to_event,
                    'event_occurred': 1 if time_to_event > 0 else 0,
                    'censored': 1 if serious == 0 else 0,
                    
                    # Patient demographics
                    'patient_age': age,
                    'patient_age_unit': age_unit,
                    'patient_sex': sex,
                    'patient_weight': weight,
                    
                    # Seriousness indicators
                    'is_serious': serious,
                    'is_death': seriousness_death,
                    'is_hospitalization': seriousness_hosp,
                    'is_lifethreatening': seriousness_life,
                    'is_disabling': seriousness_disable,
                    
                    # Drug treatment information
                    'drug_start_date': target_drug_data.get('start_date', '') if target_drug_data else '',
                    'drug_end_date': target_drug_data.get('end_date', '') if target_drug_data else '',
                    'treatment_duration': target_drug_data.get('treatment_duration', '') if target_drug_data else '',
                    'treatment_duration_unit': target_drug_data.get('treatment_duration_unit', '') if target_drug_data else '',
                    'drug_indication': target_drug_data.get('indication', '') if target_drug_data else '',
                    'administration_route': target_drug_data.get('administration_route', '') if target_drug_data else '',
                    'drug_characterization': target_drug_data.get('characterization', '') if target_drug_data else '',
                    
                    # Risk factors
                    'age_group': risk_factors['age_group'],
                    'weight_group': risk_factors['weight_group'],
                    'polypharmacy': risk_factors['polypharmacy'],
                    'multiple_events': risk_factors['multiple_events'],
                    'concomitant_drugs': risk_factors['concomitant_drugs'],
                    'drug_interaction_risk': risk_factors['drug_interaction_risk'],
                    
                    # Additional features
                    'total_drugs': len(drugs),
                    'total_events': len(adverse_events),
                    'reaction_outcome': reaction_outcomes[i] if i < len(reaction_outcomes) else '',
                    'death_date': death_date
                }
                
                survival_records.append(survival_record)
            
            return survival_records
            
        except Exception as e:
            print(f"Error processing record: {str(e)}")
            return []

    def calculate_time_to_event(self, drug_data: Optional[Dict], receive_date: str, 
                              death_date: str, serious: int) -> float:
        """
        Calculate time-to-event in days
        """
        try:
            if not drug_data:
                return 0.0
            
            drug_start = drug_data.get('start_date', '')
            drug_end = drug_data.get('end_date', '')
            
            # Parse dates
            start_date = self.parse_date(drug_start) if drug_start else None
            end_date = self.parse_date(drug_end) if drug_end else None
            report_date = self.parse_date(receive_date) if receive_date else None
            death_dt = self.parse_date(death_date) if death_date else None
            
            # Calculate time to event
            if start_date and report_date:
                if death_dt and serious:
                    # Death event - time from drug start to death
                    time_diff = (death_dt - start_date).days
                elif end_date:
                    # Drug discontinuation - time from start to end
                    time_diff = (end_date - start_date).days
                else:
                    # Report date as proxy for event
                    time_diff = (report_date - start_date).days
                
                return max(0, time_diff)
            
            return 0.0
            
        except Exception:
            return 0.0

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string to datetime object
        """
        if not date_str:
            return None
        
        try:
            # Try different date formats
            formats = [
                '%Y%m%d',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None

    def classify_adverse_event(self, ae_term: str) -> str:
        """
        Classify adverse event into categories
        """
        ae_upper = ae_term.upper()
        
        # Check for infection-related terms
        for term in self.infection_terms:
            if term in ae_upper:
                return 'INFECTION'
        
        # Check for secondary malignancy terms
        for term in self.malignancy_terms:
            if term in ae_upper:
                return 'SECONDARY_MALIGNANCY'
        
        # Check for other long-term events
        for term in self.long_term_events:
            if term in ae_upper:
                return term.replace(' ', '_')
        
        return 'OTHER'

    def extract_risk_factors(self, age: str, age_unit: str, sex: str, weight: str,
                           drug_data: List[Dict], adverse_events: List[str]) -> Dict:
        """
        Extract risk factors for survival analysis
        """
        # Age groups
        age_group = 'UNKNOWN'
        if age and age_unit:
            try:
                age_val = float(age)
                if age_unit.upper() in ['YEAR', 'YEARS', 'Y']:
                    if age_val < 18:
                        age_group = 'PEDIATRIC'
                    elif age_val < 65:
                        age_group = 'ADULT'
                    else:
                        age_group = 'ELDERLY'
                elif age_unit.upper() in ['MONTH', 'MONTHS', 'M']:
                    if age_val < 12:
                        age_group = 'INFANT'
                    else:
                        age_group = 'PEDIATRIC'
            except:
                pass
        
        # Weight groups
        weight_group = 'UNKNOWN'
        if weight:
            try:
                weight_val = float(weight)
                if weight_val < 50:
                    weight_group = 'UNDERWEIGHT'
                elif weight_val < 100:
                    weight_group = 'NORMAL'
                else:
                    weight_group = 'OVERWEIGHT'
            except:
                pass
        
        # Polypharmacy (multiple drugs)
        polypharmacy = len(drug_data) > 3
        
        # Multiple adverse events
        multiple_events = len(adverse_events) > 1
        
        # Concomitant drugs
        concomitant_drugs = len(drug_data) - 1  # Excluding target drug
        
        # Drug interaction risk (simplified)
        drug_interaction_risk = 'LOW'
        if len(drug_data) > 5:
            drug_interaction_risk = 'HIGH'
        elif len(drug_data) > 2:
            drug_interaction_risk = 'MEDIUM'
        
        return {
            'age_group': age_group,
            'weight_group': weight_group,
            'polypharmacy': polypharmacy,
            'multiple_events': multiple_events,
            'concomitant_drugs': concomitant_drugs,
            'drug_interaction_risk': drug_interaction_risk
        }


class SurvivalAnalysisModel:
    """
    Comprehensive survival analysis model for pharmacovigilance
    Implements Cox proportional hazards and other survival models
    """
    
    def __init__(self):
        self.cox_model = None
        self.km_model = None
        self.feature_importance = None
        self.risk_factors = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for survival analysis
        """
        print("Preparing data for survival analysis...")
        
        # Convert categorical variables
        categorical_cols = [
            'age_group', 'weight_group', 'drug_interaction_risk',
            'administration_route', 'drug_characterization'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Handle missing values
        numeric_cols = [
            'time_to_event_days', 'patient_age', 'patient_weight',
            'total_drugs', 'total_events', 'concomitant_drugs'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create binary indicators
        binary_cols = [
            'is_serious', 'is_death', 'is_hospitalization', 'is_lifethreatening',
            'is_disabling', 'polypharmacy', 'multiple_events', 'is_long_term_event',
            'is_infection', 'is_secondary_malignancy'
        ]
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Filter valid survival data
        df = df[df['time_to_event_days'] >= 0]
        df = df[df['time_to_event_days'] <= 3650]  # Max 10 years
        
        print(f"Prepared dataset: {len(df)} records")
        return df

    def perform_feature_selection(self, df: pd.DataFrame, target_col: str = 'is_serious') -> List[str]:
        """
        Perform feature selection to identify significant predictors
        """
        print("Performing feature selection...")
        
        # Prepare features
        feature_cols = [
            'age_group_encoded', 'weight_group_encoded', 'drug_interaction_risk_encoded',
            'administration_route_encoded', 'drug_characterization_encoded',
            'patient_age', 'patient_weight', 'total_drugs', 'total_events',
            'concomitant_drugs', 'polypharmacy', 'multiple_events'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df[target_col]
        
        # Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(10, len(available_features)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [available_features[i] for i in selector.get_support(indices=True)]
        
        # Mutual information feature selection
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(10, len(available_features)))
        mi_selector.fit(X, y)
        mi_features = [available_features[i] for i in mi_selector.get_support(indices=True)]
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Combine results
        top_features = list(set(selected_features + mi_features + rf_importance.head(5)['feature'].tolist()))
        
        print(f"Selected features: {top_features}")
        return top_features

    def fit_cox_model(self, df: pd.DataFrame, features: List[str]) -> Dict:
        """
        Fit Cox proportional hazards model
        """
        print("Fitting Cox proportional hazards model...")
        
        # Prepare data for Cox model
        cox_data = df[features + ['time_to_event_days', 'event_occurred']].copy()
        cox_data = cox_data.dropna()
        
        if len(cox_data) < 50:
            print("Insufficient data for Cox model")
            return {}
        
        # Fit Cox model
        self.cox_model = CoxPHFitter()
        self.cox_model.fit(cox_data, duration_col='time_to_event_days', event_col='event_occurred')
        
        # Get results
        cox_summary = self.cox_model.summary
        
        # Calculate concordance index
        concordance = concordance_index(
            cox_data['time_to_event_days'],
            -cox_data[features].values @ cox_summary['coef'].values,
            cox_data['event_occurred']
        )
        
        results = {
            'model': self.cox_model,
            'summary': cox_summary,
            'concordance_index': concordance,
            'n_records': len(cox_data),
            'features': features
        }
        
        print(f"Cox model fitted with concordance index: {concordance:.3f}")
        return results

    def fit_kaplan_meier(self, df: pd.DataFrame, group_col: str = 'age_group') -> Dict:
        """
        Fit Kaplan-Meier survival curves
        """
        print(f"Fitting Kaplan-Meier curves by {group_col}...")
        
        # Prepare data
        km_data = df[['time_to_event_days', 'event_occurred', group_col]].copy()
        km_data = km_data.dropna()
        
        if len(km_data) < 20:
            print("Insufficient data for Kaplan-Meier analysis")
            return {}
        
        # Fit Kaplan-Meier for each group
        groups = km_data[group_col].unique()
        km_results = {}
        
        for group in groups:
            group_data = km_data[km_data[group_col] == group]
            if len(group_data) >= 10:
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['time_to_event_days'], group_data['event_occurred'])
                km_results[group] = kmf
        
        # Perform log-rank test
        if len(groups) > 1:
            group_data_list = []
            for group in groups:
                group_data = km_data[km_data[group_col] == group]
                if len(group_data) >= 10:
                    group_data_list.append(group_data)
            
            if len(group_data_list) >= 2:
                logrank_result = multivariate_logrank_test(
                    *[group['time_to_event_days'] for group in group_data_list],
                    *[group['event_occurred'] for group in group_data_list]
                )
                
                km_results['logrank_pvalue'] = logrank_result.p_value
                km_results['logrank_statistic'] = logrank_result.test_statistic
        
        return km_results

    def analyze_long_term_events(self, df: pd.DataFrame) -> Dict:
        """
        Analyze long-term adverse events (infections, secondary malignancies)
        """
        print("Analyzing long-term adverse events...")
        
        # Filter for long-term events
        long_term_df = df[df['is_long_term_event'] == 1].copy()
        
        if len(long_term_df) < 20:
            print("Insufficient long-term event data")
            return {}
        
        # Infection analysis
        infection_df = df[df['is_infection'] == 1].copy()
        infection_analysis = {}
        
        if len(infection_df) >= 10:
            # Time to infection
            infection_times = infection_df['time_to_event_days'].values
            infection_analysis = {
                'n_infections': len(infection_df),
                'median_time_to_infection': np.median(infection_times),
                'mean_time_to_infection': np.mean(infection_times),
                'infection_rate': len(infection_df) / len(df),
                'serious_infection_rate': infection_df['is_serious'].mean(),
                'death_from_infection': infection_df['is_death'].mean()
            }
        
        # Secondary malignancy analysis
        malignancy_df = df[df['is_secondary_malignancy'] == 1].copy()
        malignancy_analysis = {}
        
        if len(malignancy_df) >= 10:
            malignancy_times = malignancy_df['time_to_event_days'].values
            malignancy_analysis = {
                'n_malignancies': len(malignancy_df),
                'median_time_to_malignancy': np.median(malignancy_times),
                'mean_time_to_malignancy': np.mean(malignancy_times),
                'malignancy_rate': len(malignancy_df) / len(df),
                'serious_malignancy_rate': malignancy_df['is_serious'].mean(),
                'death_from_malignancy': malignancy_df['is_death'].mean()
            }
        
        # Risk factor analysis for long-term events
        risk_factors = self.identify_long_term_risk_factors(df)
        
        return {
            'infection_analysis': infection_analysis,
            'malignancy_analysis': malignancy_analysis,
            'risk_factors': risk_factors,
            'total_long_term_events': len(long_term_df),
            'long_term_event_rate': len(long_term_df) / len(df)
        }

    def identify_long_term_risk_factors(self, df: pd.DataFrame) -> Dict:
        """
        Identify risk factors for long-term adverse events
        """
        print("Identifying risk factors for long-term events...")
        
        # Prepare features
        feature_cols = [
            'age_group_encoded', 'weight_group_encoded', 'drug_interaction_risk_encoded',
            'patient_age', 'patient_weight', 'total_drugs', 'total_events',
            'concomitant_drugs', 'polypharmacy', 'multiple_events'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) == 0:
            return {}
        
        X = df[available_features].fillna(0)
        y = df['is_long_term_event']
        
        # Logistic regression for risk factors
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, y)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'feature': available_features,
            'coefficient': lr.coef_[0],
            'odds_ratio': np.exp(lr.coef_[0])
        }).sort_values('coefficient', key=abs, ascending=False)
        
        # Cross-validation score
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
        
        return {
            'coefficients': coefficients,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'model': lr
        }

    def generate_risk_predictions(self, df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """
        Generate risk predictions for new patients
        """
        print("Generating risk predictions...")
        
        if not model_results or 'model' not in model_results:
            print("No trained model available")
            return df
        
        model = model_results['model']
        features = model_results['features']
        
        # Prepare features
        X = df[features].fillna(0)
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            risk_scores = model.predict_proba(X)[:, 1]
        else:
            risk_scores = model.predict(X)
        
        # Add predictions to dataframe
        df['risk_score'] = risk_scores
        df['risk_category'] = pd.cut(risk_scores, bins=3, labels=['LOW', 'MEDIUM', 'HIGH'])
        
        return df

    def create_visualizations(self, df: pd.DataFrame, results: Dict, output_dir: str = './'):
        """
        Create comprehensive visualizations for survival analysis
        """
        print("Creating visualizations...")
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Kaplan-Meier curves
        if 'kaplan_meier' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            km_results = results['kaplan_meier']
            
            for group, kmf in km_results.items():
                if isinstance(group, str) and group not in ['logrank_pvalue', 'logrank_statistic']:
                    kmf.plot_survival_function(ax=ax, label=f'{group}')
            
            ax.set_title('Kaplan-Meier Survival Curves')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Survival Probability')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/kaplan_meier_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Risk factor importance
        if 'cox_model' in results and results['cox_model']:
            fig, ax = plt.subplots(figsize=(10, 6))
            cox_summary = results['cox_model']['summary']
            
            # Plot hazard ratios
            cox_summary['hazard_ratio'] = np.exp(cox_summary['coef'])
            cox_summary = cox_summary.sort_values('hazard_ratio', ascending=True)
            
            y_pos = np.arange(len(cox_summary))
            ax.barh(y_pos, cox_summary['hazard_ratio'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cox_summary.index)
            ax.set_xlabel('Hazard Ratio')
            ax.set_title('Cox Model: Hazard Ratios for Risk Factors')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cox_hazard_ratios.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Long-term event analysis
        if 'long_term_analysis' in results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Infection vs malignancy rates
            infection_data = results['long_term_analysis'].get('infection_analysis', {})
            malignancy_data = results['long_term_analysis'].get('malignancy_analysis', {})
            
            if infection_data and malignancy_data:
                categories = ['Infection Rate', 'Malignancy Rate']
                rates = [
                    infection_data.get('infection_rate', 0),
                    malignancy_data.get('malignancy_rate', 0)
                ]
                
                axes[0, 0].bar(categories, rates)
                axes[0, 0].set_title('Long-term Event Rates')
                axes[0, 0].set_ylabel('Rate')
            
            # Time to events
            if infection_data and malignancy_data:
                times = [
                    infection_data.get('median_time_to_infection', 0),
                    malignancy_data.get('median_time_to_malignancy', 0)
                ]
                
                axes[0, 1].bar(categories, times)
                axes[0, 1].set_title('Median Time to Events')
                axes[0, 1].set_ylabel('Days')
            
            # Risk factor importance
            if 'risk_factors' in results['long_term_analysis']:
                risk_factors = results['long_term_analysis']['risk_factors']
                if 'coefficients' in risk_factors:
                    coef_df = risk_factors['coefficients'].head(10)
                    
                    axes[1, 0].barh(range(len(coef_df)), coef_df['coefficient'])
                    axes[1, 0].set_yticks(range(len(coef_df)))
                    axes[1, 0].set_yticklabels(coef_df['feature'])
                    axes[1, 0].set_title('Risk Factor Coefficients')
                    axes[1, 0].set_xlabel('Coefficient')
            
            # Risk score distribution
            if 'risk_score' in df.columns:
                axes[1, 1].hist(df['risk_score'], bins=30, alpha=0.7)
                axes[1, 1].set_title('Risk Score Distribution')
                axes[1, 1].set_xlabel('Risk Score')
                axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/long_term_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function to run comprehensive survival analysis
    """
    print("=" * 80)
    print("Requirement 2: Model Risk Factors and Time-to-Event Analysis")
    print("AI-Powered Pharmacovigilance System")
    print("=" * 80)
    
    # Initialize components
    collector = SurvivalDataCollector()
    model = SurvivalAnalysisModel()
    
    # Collect data for key oncology drugs
    key_drugs = ["Epcoritamab", "Pembrolizumab", "Nivolumab", "Trastuzumab", "Rituximab"]
    all_data = []
    
    for drug in key_drugs:
        print(f"\nCollecting data for {drug}...")
        drug_data = collector.collect_survival_data(drug, limit=500)
        all_data.extend(drug_data)
        
        # Save intermediate results
        if len(all_data) > 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_csv(f'survival_data_temp_{drug}.csv', index=False)
            print(f"Saved {len(all_data)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("No data collected. Exiting.")
        return
    
    # Data preparation
    df = model.prepare_data(df)
    
    # Feature selection
    features = model.perform_feature_selection(df)
    
    # Survival analysis
    results = {}
    
    # Cox proportional hazards model
    cox_results = model.fit_cox_model(df, features)
    if cox_results:
        results['cox_model'] = cox_results
    
    # Kaplan-Meier analysis
    km_results = model.fit_kaplan_meier(df, 'age_group')
    if km_results:
        results['kaplan_meier'] = km_results
    
    # Long-term event analysis
    long_term_results = model.analyze_long_term_events(df)
    if long_term_results:
        results['long_term_analysis'] = long_term_results
    
    # Generate predictions
    df_with_predictions = model.generate_risk_predictions(df, cox_results)
    
    # Create visualizations
    model.create_visualizations(df_with_predictions, results)
    
    # Save results
    df_with_predictions.to_csv('requirement2_survival_analysis_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SURVIVAL ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total records analyzed: {len(df)}")
    print(f"Features selected: {len(features)}")
    
    if 'cox_model' in results:
        print(f"Cox model concordance index: {results['cox_model']['concordance_index']:.3f}")
    
    if 'long_term_analysis' in results:
        lt_analysis = results['long_term_analysis']
        print(f"Long-term event rate: {lt_analysis.get('long_term_event_rate', 0):.3f}")
        
        if 'infection_analysis' in lt_analysis:
            inf_analysis = lt_analysis['infection_analysis']
            print(f"Infection rate: {inf_analysis.get('infection_rate', 0):.3f}")
            print(f"Median time to infection: {inf_analysis.get('median_time_to_infection', 0):.1f} days")
        
        if 'malignancy_analysis' in lt_analysis:
            mal_analysis = lt_analysis['malignancy_analysis']
            print(f"Secondary malignancy rate: {mal_analysis.get('malignancy_rate', 0):.3f}")
            print(f"Median time to malignancy: {mal_analysis.get('median_time_to_malignancy', 0):.1f} days")
    
    print("\nResults saved to:")
    print("- requirement2_survival_analysis_results.csv")
    print("- kaplan_meier_curves.png")
    print("- cox_hazard_ratios.png")
    print("- long_term_analysis.png")


if __name__ == "__main__":
    main()
