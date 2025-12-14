#!/usr/bin/env python3
"""
Requirement 2: Simplified Demo Script
AI-Powered Pharmacovigilance System - Survival Analysis Demo

This script demonstrates the key functionality of Requirement 2
with a simplified approach to avoid convergence issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import basic libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def create_sample_data(n_records=1000):
    """
    Create sample data for demonstration purposes
    """
    print("Creating sample data for demonstration...")
    
    np.random.seed(42)
    
    # 35种常见肿瘤药物（聚焦靶向治疗和免疫治疗）
    drugs = [
        "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",
        "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab", "Epcoritamab",
        "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
        "Paclitaxel", "Docetaxel", "Doxorubicin", "Carboplatin", "Cisplatin",
        "Lenalidomide", "Pomalidomide", "Bortezomib", "Carfilzomib", "Venetoclax",
        "Ibrutinib", "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
        "Palbociclib", "Ribociclib", "Abemaciclib", "Vemurafenib", "Dabrafenib"
    ]
    
    # Sample adverse events
    adverse_events = [
        'INFECTION', 'SECONDARY MALIGNANCY', 'CARDIOVASCULAR DISORDER',
        'NEUROLOGICAL DISORDER', 'RENAL DISORDER', 'HEPATIC DISORDER',
        'PULMONARY DISORDER', 'ENDOCRINE DISORDER', 'IMMUNE DISORDER',
        'METABOLIC DISORDER', 'SKELETAL DISORDER', 'DERMATOLOGICAL DISORDER',
        'GASTROINTESTINAL DISORDER', 'HEMATOLOGICAL DISORDER'
    ]
    
    data = []
    
    for i in range(n_records):
        # Basic information
        drug = np.random.choice(drugs)
        ae = np.random.choice(adverse_events)
        
        # Patient demographics
        age = np.random.normal(65, 15)
        age = max(18, min(90, age))  # Clamp to reasonable range
        
        age_group = 'PEDIATRIC' if age < 18 else 'ADULT' if age < 65 else 'ELDERLY'
        sex = np.random.choice([1, 2])  # 1=Male, 2=Female
        weight = np.random.normal(70, 15)
        
        # Drug treatment
        drug_start_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
        time_to_event = np.random.exponential(60)  # Exponential distribution for survival times
        
        # Event characteristics
        is_serious = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% serious
        is_long_term = 1 if ae in ['INFECTION', 'SECONDARY MALIGNANCY'] else 0
        is_infection = 1 if ae == 'INFECTION' else 0
        is_malignancy = 1 if ae == 'SECONDARY MALIGNANCY' else 0
        
        # Risk factors
        polypharmacy = np.random.choice([True, False], p=[0.4, 0.6])
        total_drugs = np.random.poisson(3) + 1
        multiple_events = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Create record
        record = {
            'safety_report_id': f'SR{i:06d}',
            'receive_date': (drug_start_date + timedelta(days=time_to_event)).strftime('%Y%m%d'),
            'target_drug': drug,
            'adverse_event': ae,
            'event_type': ae.replace(' ', '_'),
            'is_long_term_event': is_long_term,
            'is_infection': is_infection,
            'is_secondary_malignancy': is_malignancy,
            
            # Time-to-event data
            'time_to_event_days': time_to_event,
            'event_occurred': 1,
            'censored': 0,
            
            # Patient demographics
            'patient_age': age,
            'patient_age_unit': 'YEAR',
            'patient_sex': sex,
            'patient_weight': weight,
            
            # Seriousness indicators
            'is_serious': is_serious,
            'is_death': np.random.choice([0, 1], p=[0.95, 0.05]) if is_serious else 0,
            'is_hospitalization': np.random.choice([0, 1], p=[0.7, 0.3]) if is_serious else 0,
            'is_lifethreatening': np.random.choice([0, 1], p=[0.8, 0.2]) if is_serious else 0,
            'is_disabling': np.random.choice([0, 1], p=[0.9, 0.1]) if is_serious else 0,
            
            # Risk factors
            'age_group': age_group,
            'weight_group': 'UNDERWEIGHT' if weight < 50 else 'NORMAL' if weight < 100 else 'OVERWEIGHT',
            'polypharmacy': polypharmacy,
            'multiple_events': multiple_events,
            'concomitant_drugs': max(0, total_drugs - 1),
            'drug_interaction_risk': 'LOW' if total_drugs <= 2 else 'MEDIUM' if total_drugs <= 5 else 'HIGH',
            
            # Additional features
            'total_drugs': total_drugs,
            'total_events': np.random.poisson(2) + 1,
            'administration_route': np.random.choice(['INTRAVENOUS', 'SUBCUTANEOUS', 'ORAL']),
            'drug_characterization': np.random.choice([1, 2, 3])  # 1=Suspect, 2=Concomitant, 3=Interacting
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    print(f"Created {len(df)} sample records")
    print(f"Drugs: {df['target_drug'].nunique()}")
    print(f"Adverse events: {df['adverse_event'].nunique()}")
    print(f"Serious events: {df['is_serious'].sum()} ({df['is_serious'].mean()*100:.1f}%)")
    print(f"Long-term events: {df['is_long_term_event'].sum()} ({df['is_long_term_event'].mean()*100:.1f}%)")
    
    return df


def run_simplified_analysis(df):
    """
    Run simplified survival analysis using Random Forest
    """
    print("\n" + "=" * 80)
    print("SIMPLIFIED SURVIVAL ANALYSIS")
    print("=" * 80)
    
    # Prepare features
    categorical_cols = ['age_group', 'weight_group', 'drug_interaction_risk', 'administration_route']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Select features
    feature_cols = [
        'age_group_encoded', 'weight_group_encoded', 'drug_interaction_risk_encoded',
        'administration_route_encoded', 'patient_age', 'patient_weight', 'total_drugs',
        'total_events', 'concomitant_drugs', 'polypharmacy', 'multiple_events',
        'drug_characterization'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['is_serious']
    
    print(f"\nFeatures used: {available_features}")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"AUC Score: {auc_score:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Long-term event analysis
    print(f"\nLong-term Event Analysis:")
    long_term_df = df[df['is_long_term_event'] == 1]
    
    if len(long_term_df) > 0:
        print(f"  Long-term event rate: {len(long_term_df)/len(df):.3f}")
        print(f"  Median time to long-term events: {long_term_df['time_to_event_days'].median():.1f} days")
        
        # Infection analysis
        infection_df = df[df['is_infection'] == 1]
        if len(infection_df) > 0:
            print(f"  Infection rate: {len(infection_df)/len(df):.3f}")
            print(f"  Median time to infection: {infection_df['time_to_event_days'].median():.1f} days")
        
        # Malignancy analysis
        malignancy_df = df[df['is_secondary_malignancy'] == 1]
        if len(malignancy_df) > 0:
            print(f"  Secondary malignancy rate: {len(malignancy_df)/len(df):.3f}")
            print(f"  Median time to malignancy: {malignancy_df['time_to_event_days'].median():.1f} days")
    
    # Age-related patterns
    print(f"\nAge-related Risk Patterns:")
    age_patterns = df.groupby('age_group').agg({
        'is_serious': 'mean',
        'is_long_term_event': 'mean',
        'time_to_event_days': 'median'
    }).round(3)
    
    for age_group, row in age_patterns.iterrows():
        print(f"  {age_group}: Serious={row['is_serious']:.3f}, Long-term={row['is_long_term_event']:.3f}, Median time={row['time_to_event_days']:.1f} days")
    
    # Drug-specific patterns
    print(f"\nDrug-specific Safety Profiles:")
    drug_patterns = df.groupby('target_drug').agg({
        'is_serious': 'mean',
        'is_long_term_event': 'mean',
        'time_to_event_days': 'median'
    }).round(3)
    
    for drug, row in drug_patterns.iterrows():
        print(f"  {drug}: Serious={row['is_serious']:.3f}, Long-term={row['is_long_term_event']:.3f}, Median time={row['time_to_event_days']:.1f} days")
    
    # Create visualizations
    create_visualizations(df, feature_importance, output_dir='./')
    
    return df, rf, feature_importance


def create_visualizations(df, feature_importance, output_dir='./'):
    """
    Create visualizations for the analysis
    """
    print(f"\nCreating visualizations...")
    
    plt.style.use('default')
    
    # 1. Feature importance plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top 10 features
    top_features = feature_importance.head(10)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    
    # Age group patterns
    age_patterns = df.groupby('age_group')['is_serious'].mean()
    axes[0, 1].bar(age_patterns.index, age_patterns.values)
    axes[0, 1].set_title('Serious Event Rate by Age Group')
    axes[0, 1].set_ylabel('Serious Event Rate')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Time to event distribution
    axes[1, 0].hist(df['time_to_event_days'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Time to Event Distribution')
    axes[1, 0].set_xlabel('Days')
    axes[1, 0].set_ylabel('Frequency')
    
    # Long-term vs other events
    lt_df = df[df['is_long_term_event'] == 1]
    other_df = df[df['is_long_term_event'] == 0]
    
    if len(lt_df) > 0 and len(other_df) > 0:
        axes[1, 1].hist([lt_df['time_to_event_days'], other_df['time_to_event_days']], 
                       bins=20, alpha=0.7, label=['Long-term Events', 'Other Events'])
        axes[1, 1].set_title('Time to Event: Long-term vs Other')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/requirement2_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}/requirement2_analysis_summary.png")


def save_results(df, feature_importance, output_dir='./'):
    """
    Save analysis results to files
    """
    print(f"\nSaving results...")
    
    # Save data
    df.to_csv(f'{output_dir}/requirement2_sample_data.csv', index=False)
    feature_importance.to_csv(f'{output_dir}/requirement2_feature_importance.csv', index=False)
    
    # Create summary report
    with open(f'{output_dir}/requirement2_summary_report.txt', 'w') as f:
        f.write("REQUIREMENT 2: SURVIVAL ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"- Total records: {len(df):,}\n")
        f.write(f"- Drugs analyzed: {df['target_drug'].nunique()}\n")
        f.write(f"- Adverse events: {df['adverse_event'].nunique()}\n")
        f.write(f"- Serious events: {df['is_serious'].sum():,} ({df['is_serious'].mean()*100:.1f}%)\n")
        f.write(f"- Long-term events: {df['is_long_term_event'].sum():,} ({df['is_long_term_event'].mean()*100:.1f}%)\n\n")
        
        f.write(f"Top Risk Factors:\n")
        for i, row in feature_importance.head(10).iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.3f}\n")
        
        f.write(f"\nLong-term Event Analysis:\n")
        long_term_df = df[df['is_long_term_event'] == 1]
        if len(long_term_df) > 0:
            f.write(f"- Long-term event rate: {len(long_term_df)/len(df):.3f}\n")
            f.write(f"- Median time to long-term events: {long_term_df['time_to_event_days'].median():.1f} days\n")
        
        infection_df = df[df['is_infection'] == 1]
        if len(infection_df) > 0:
            f.write(f"- Infection rate: {len(infection_df)/len(df):.3f}\n")
            f.write(f"- Median time to infection: {infection_df['time_to_event_days'].median():.1f} days\n")
        
        malignancy_df = df[df['is_secondary_malignancy'] == 1]
        if len(malignancy_df) > 0:
            f.write(f"- Secondary malignancy rate: {len(malignancy_df)/len(df):.3f}\n")
            f.write(f"- Median time to malignancy: {malignancy_df['time_to_event_days'].median():.1f} days\n")
    
    print(f"✓ Results saved to:")
    print(f"  - {output_dir}/requirement2_sample_data.csv")
    print(f"  - {output_dir}/requirement2_feature_importance.csv")
    print(f"  - {output_dir}/requirement2_summary_report.txt")
    print(f"  - {output_dir}/requirement2_analysis_summary.png")


def main():
    """
    Main function to run simplified demonstration
    """
    print("=" * 80)
    print("REQUIREMENT 2: SIMPLIFIED DEMONSTRATION")
    print("AI-Powered Pharmacovigilance System")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data(n_records=1000)
    
    # Run analysis
    df_analyzed, model, feature_importance = run_simplified_analysis(df)
    
    # Save results
    save_results(df_analyzed, feature_importance)
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print(f"✓ Total records analyzed: {len(df_analyzed):,}")
    print(f"✓ Features analyzed: {len(feature_importance)}")
    print(f"✓ Long-term event rate: {df_analyzed['is_long_term_event'].mean():.3f}")
    print(f"✓ Serious event rate: {df_analyzed['is_serious'].mean():.3f}")
    
    print(f"\nFiles generated:")
    print(f"  - requirement2_sample_data.csv")
    print(f"  - requirement2_feature_importance.csv") 
    print(f"  - requirement2_summary_report.txt")
    print(f"  - requirement2_analysis_summary.png")
    
    print(f"\nThis demonstration shows the core functionality of Requirement 2")
    print(f"using simplified methods. For full Cox survival analysis with real")
    print(f"FAERS data, run: python3 requirement2_integration.py")


if __name__ == "__main__":
    main()
