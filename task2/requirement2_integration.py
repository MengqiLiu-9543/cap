#!/usr/bin/env python3
"""
Requirement 2: Complete Integration Script
AI-Powered Pharmacovigilance System - Survival Analysis Integration

This script integrates all components of Requirement 2:
1. Data collection for survival analysis
2. Cox proportional hazards modeling
3. Feature selection and risk factor identification
4. Long-term adverse event analysis
5. Comprehensive validation and testing
6. Report generation and visualization

Usage:
    python requirement2_integration.py [--drugs DRUG1,DRUG2] [--limit LIMIT] [--output-dir DIR]
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from requirement2_survival_analysis import SurvivalDataCollector, SurvivalAnalysisModel
from requirement2_validation import run_comprehensive_validation, create_validation_plots


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Requirement 2: Model Risk Factors and Time-to-Event Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default oncology drugs
    python requirement2_integration.py
    
    # Run with specific drugs
    python requirement2_integration.py --drugs "Epcoritamab,Pembrolizumab,Nivolumab"
    
    # Run with custom limit and output directory
    python requirement2_integration.py --limit 1000 --output-dir ./results/
        """
    )
    
    parser.add_argument(
        '--drugs',
        type=str,
        default='Pembrolizumab,Nivolumab,Atezolizumab,Durvalumab,Ipilimumab,Trastuzumab,Bevacizumab,Cetuximab,Rituximab,Epcoritamab,Imatinib,Erlotinib,Gefitinib,Osimertinib,Crizotinib,Paclitaxel,Docetaxel,Doxorubicin,Carboplatin,Cisplatin,Lenalidomide,Pomalidomide,Bortezomib,Carfilzomib,Venetoclax,Ibrutinib,Olaparib,Rucaparib,Niraparib,Talazoparib,Palbociclib,Ribociclib,Abemaciclib,Vemurafenib,Dabrafenib',
        help='Comma-separated list of oncology drugs to analyze (default: key oncology drugs)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=500,
        help='Maximum number of records to collect per drug (default: 500)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./requirement2_results/',
        help='Output directory for results (default: ./requirement2_results/)'
    )
    
    parser.add_argument(
        '--skip-data-collection',
        action='store_true',
        help='Skip data collection and use existing data files'
    )
    
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help='Run only validation on existing results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """
    Create and setup output directory structure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'data').mkdir(exist_ok=True)
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'reports').mkdir(exist_ok=True)
    
    return output_path


def collect_survival_data(drugs: list, limit: int, output_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """
    Collect comprehensive survival analysis data
    """
    print("=" * 80)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 80)
    
    collector = SurvivalDataCollector()
    all_data = []
    
    for i, drug in enumerate(drugs, 1):
        print(f"\n[{i}/{len(drugs)}] Collecting data for {drug}...")
        
        try:
            drug_data = collector.collect_survival_data(drug, limit=limit)
            all_data.extend(drug_data)
            
            # Save intermediate results
            if drug_data:
                temp_df = pd.DataFrame(drug_data)
                temp_file = output_dir / 'data' / f'survival_data_{drug}.csv'
                temp_df.to_csv(temp_file, index=False)
                
                if verbose:
                    print(f"  ✓ Saved {len(drug_data)} records to {temp_file}")
            
        except Exception as e:
            print(f"  ✗ Error collecting data for {drug}: {str(e)}")
            continue
    
    # Combine all data
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save combined dataset
        combined_file = output_dir / 'data' / 'combined_survival_data.csv'
        df.to_csv(combined_file, index=False)
        
        print(f"\n✓ Data collection complete: {len(df)} total records")
        print(f"✓ Combined dataset saved to: {combined_file}")
        
        return df
    else:
        print("\n✗ No data collected. Exiting.")
        sys.exit(1)


def run_survival_analysis(df: pd.DataFrame, output_dir: Path, verbose: bool = False) -> dict:
    """
    Run comprehensive survival analysis
    """
    print("\n" + "=" * 80)
    print("PHASE 2: SURVIVAL ANALYSIS")
    print("=" * 80)
    
    model = SurvivalAnalysisModel()
    results = {}
    
    # 1. Data preparation
    print("\n1. Preparing data for survival analysis...")
    df_prepared = model.prepare_data(df.copy())
    
    if verbose:
        print(f"   Prepared dataset: {len(df_prepared)} records")
        print(f"   Features available: {len(df_prepared.columns)}")
    
    # 2. Feature selection
    print("\n2. Performing feature selection...")
    features = model.perform_feature_selection(df_prepared)
    
    if verbose:
        print(f"   Selected features: {features}")
    
    # 3. Cox proportional hazards model
    print("\n3. Fitting Cox proportional hazards model...")
    cox_results = model.fit_cox_model(df_prepared, features)
    
    if cox_results:
        results['cox_model'] = cox_results
        
        if verbose:
            print(f"   ✓ Cox model fitted successfully")
            print(f"   ✓ Concordance index: {cox_results['concordance_index']:.3f}")
    
    # 4. Kaplan-Meier analysis
    print("\n4. Performing Kaplan-Meier analysis...")
    km_results = model.fit_kaplan_meier(df_prepared, 'age_group')
    
    if km_results:
        results['kaplan_meier'] = km_results
        
        if verbose:
            print(f"   ✓ Kaplan-Meier curves generated for {len(km_results)-2} groups")
    
    # 5. Long-term event analysis
    print("\n5. Analyzing long-term adverse events...")
    long_term_results = model.analyze_long_term_events(df_prepared)
    
    if long_term_results:
        results['long_term_analysis'] = long_term_results
        
        if verbose:
            lt_rate = long_term_results.get('long_term_event_rate', 0)
            print(f"   ✓ Long-term event rate: {lt_rate:.3f}")
            
            if 'infection_analysis' in long_term_results:
                inf_rate = long_term_results['infection_analysis'].get('infection_rate', 0)
                print(f"   ✓ Infection rate: {inf_rate:.3f}")
            
            if 'malignancy_analysis' in long_term_results:
                mal_rate = long_term_results['malignancy_analysis'].get('malignancy_rate', 0)
                print(f"   ✓ Secondary malignancy rate: {mal_rate:.3f}")
    
    # 6. Generate risk predictions
    print("\n6. Generating risk predictions...")
    df_with_predictions = model.generate_risk_predictions(df_prepared, cox_results)
    
    # Save results
    results_file = output_dir / 'data' / 'survival_analysis_results.csv'
    df_with_predictions.to_csv(results_file, index=False)
    
    if verbose:
        print(f"   ✓ Results saved to: {results_file}")
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    model.create_visualizations(df_with_predictions, results, str(output_dir / 'plots'))
    
    if verbose:
        print(f"   ✓ Visualizations saved to: {output_dir / 'plots'}")
    
    return results, df_with_predictions


def run_validation_and_testing(df: pd.DataFrame, model_results: dict, output_dir: Path, verbose: bool = False) -> dict:
    """
    Run comprehensive validation and testing
    """
    print("\n" + "=" * 80)
    print("PHASE 3: VALIDATION AND TESTING")
    print("=" * 80)
    
    # Run comprehensive validation
    validation_results = run_comprehensive_validation(df, model_results)
    
    # Save validation results
    validation_file = output_dir / 'reports' / 'validation_results.json'
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    validation_json = convert_numpy(validation_results)
    
    with open(validation_file, 'w') as f:
        json.dump(validation_json, f, indent=2)
    
    if verbose:
        print(f"✓ Validation results saved to: {validation_file}")
    
    return validation_results


def generate_final_report(df: pd.DataFrame, model_results: dict, validation_results: dict, 
                         output_dir: Path, drugs: list) -> None:
    """
    Generate comprehensive final report
    """
    print("\n" + "=" * 80)
    print("PHASE 4: FINAL REPORT GENERATION")
    print("=" * 80)
    
    report_file = output_dir / 'reports' / 'requirement2_final_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Requirement 2: Model Risk Factors and Time-to-Event Analysis\n")
        f.write("AI-Powered Pharmacovigilance System\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of comprehensive survival analysis and risk factor modeling ")
        f.write("for oncology drug safety monitoring. The analysis focuses on time-to-event modeling using ")
        f.write("Cox proportional hazards regression and identifies significant predictors of serious adverse events.\n\n")
        
        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Records Analyzed:** {len(df):,}\n")
        f.write(f"- **Drugs Analyzed:** {', '.join(drugs)}\n")
        f.write(f"- **Unique Patients:** {df['safety_report_id'].nunique():,}\n")
        f.write(f"- **Adverse Events:** {df['adverse_event'].nunique():,}\n")
        f.write(f"- **Serious Events:** {df['is_serious'].sum():,} ({df['is_serious'].mean()*100:.1f}%)\n")
        f.write(f"- **Long-term Events:** {df['is_long_term_event'].sum():,} ({df['is_long_term_event'].mean()*100:.1f}%)\n\n")
        
        # Model Performance
        f.write("## Model Performance\n\n")
        
        if 'cox_model' in model_results:
            cox_results = model_results['cox_model']
            f.write("### Cox Proportional Hazards Model\n\n")
            f.write(f"- **Concordance Index:** {cox_results.get('concordance_index', 'N/A'):.3f}\n")
            f.write(f"- **Number of Records:** {cox_results.get('n_records', 'N/A'):,}\n")
            f.write(f"- **Features Used:** {len(cox_results.get('features', []))}\n\n")
            
            # Feature importance
            if 'summary' in cox_results:
                f.write("#### Top Risk Factors (Hazard Ratios)\n\n")
                summary = cox_results['summary']
                summary_sorted = summary.sort_values('coef', key=abs, ascending=False).head(10)
                
                for feature, row in summary_sorted.iterrows():
                    hr = np.exp(row['coef'])
                    f.write(f"- **{feature}:** HR = {hr:.3f} (95% CI: {np.exp(row['coef'] - 1.96*row['se']):.3f} - {np.exp(row['coef'] + 1.96*row['se']):.3f})\n")
                f.write("\n")
        
        # Long-term Event Analysis
        if 'long_term_analysis' in model_results:
            f.write("### Long-term Adverse Events Analysis\n\n")
            lt_analysis = model_results['long_term_analysis']
            
            f.write(f"- **Overall Long-term Event Rate:** {lt_analysis.get('long_term_event_rate', 0):.3f}\n")
            
            if 'infection_analysis' in lt_analysis:
                inf_analysis = lt_analysis['infection_analysis']
                f.write(f"- **Infection Rate:** {inf_analysis.get('infection_rate', 0):.3f}\n")
                f.write(f"- **Median Time to Infection:** {inf_analysis.get('median_time_to_infection', 0):.1f} days\n")
                f.write(f"- **Serious Infection Rate:** {inf_analysis.get('serious_infection_rate', 0):.3f}\n")
            
            if 'malignancy_analysis' in lt_analysis:
                mal_analysis = lt_analysis['malignancy_analysis']
                f.write(f"- **Secondary Malignancy Rate:** {mal_analysis.get('malignancy_rate', 0):.3f}\n")
                f.write(f"- **Median Time to Malignancy:** {mal_analysis.get('median_time_to_malignancy', 0):.1f} days\n")
                f.write(f"- **Serious Malignancy Rate:** {mal_analysis.get('serious_malignancy_rate', 0):.3f}\n")
            
            f.write("\n")
        
        # Validation Results
        f.write("## Validation Results\n\n")
        
        if 'cox_model' in validation_results:
            cox_val = validation_results['cox_model']
            f.write("### Cox Model Validation\n\n")
            
            if 'concordance_index' in cox_val:
                f.write(f"- **Concordance Index:** {cox_val['concordance_index']:.3f}\n")
            
            if 'c_index_ci' in cox_val and cox_val['c_index_ci'].get('mean'):
                ci = cox_val['c_index_ci']
                f.write(f"- **Bootstrap C-index CI:** [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]\n")
            
            if 'cross_validation' in cox_val and cox_val['cross_validation'].get('mean_c_index'):
                cv = cox_val['cross_validation']
                f.write(f"- **Cross-validation C-index:** {cv['mean_c_index']:.3f} ± {cv['std_c_index']:.3f}\n")
            
            f.write("\n")
        
        # Clinical Insights
        f.write("## Clinical Insights\n\n")
        
        # Age-related patterns
        age_patterns = df.groupby('age_group').agg({
            'is_serious': 'mean',
            'is_long_term_event': 'mean',
            'time_to_event_days': 'median'
        }).round(3)
        
        f.write("### Age-related Risk Patterns\n\n")
        f.write("| Age Group | Serious Event Rate | Long-term Event Rate | Median Time to Event (days) |\n")
        f.write("|-----------|-------------------|---------------------|------------------------------|\n")
        
        for age_group, row in age_patterns.iterrows():
            f.write(f"| {age_group} | {row['is_serious']:.3f} | {row['is_long_term_event']:.3f} | {row['time_to_event_days']:.1f} |\n")
        
        f.write("\n")
        
        # Drug-specific insights
        f.write("### Drug-specific Safety Profiles\n\n")
        drug_patterns = df.groupby('target_drug').agg({
            'is_serious': 'mean',
            'is_long_term_event': 'mean',
            'is_infection': 'mean',
            'is_secondary_malignancy': 'mean',
            'time_to_event_days': 'median'
        }).round(3)
        
        f.write("| Drug | Serious Rate | Long-term Rate | Infection Rate | Malignancy Rate | Median Time (days) |\n")
        f.write("|------|-------------|----------------|----------------|-----------------|-------------------|\n")
        
        for drug, row in drug_patterns.iterrows():
            f.write(f"| {drug} | {row['is_serious']:.3f} | {row['is_long_term_event']:.3f} | {row['is_infection']:.3f} | {row['is_secondary_malignancy']:.3f} | {row['time_to_event_days']:.1f} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        f.write("### For Clinical Practice\n")
        f.write("- Monitor elderly patients more closely for long-term adverse events\n")
        f.write("- Consider polypharmacy as a risk factor for serious events\n")
        f.write("- Implement extended follow-up for patients on high-risk oncology drugs\n\n")
        
        f.write("### For Future Research\n")
        f.write("- Validate findings in larger, prospective cohorts\n")
        f.write("- Investigate drug-drug interactions in polypharmacy patients\n")
        f.write("- Develop personalized risk prediction models\n\n")
        
        # Technical Details
        f.write("## Technical Details\n\n")
        f.write("### Data Sources\n")
        f.write("- FDA Adverse Event Reporting System (FAERS)\n")
        f.write("- OpenFDA API (https://open.fda.gov/apis/)\n\n")
        
        f.write("### Methodology\n")
        f.write("- Cox proportional hazards regression\n")
        f.write("- Kaplan-Meier survival analysis\n")
        f.write("- Feature selection using multiple methods\n")
        f.write("- Bootstrap validation\n")
        f.write("- Cross-validation for model assessment\n\n")
        
        f.write("### Files Generated\n")
        f.write("- `combined_survival_data.csv`: Raw collected data\n")
        f.write("- `survival_analysis_results.csv`: Processed data with predictions\n")
        f.write("- `validation_results.json`: Comprehensive validation results\n")
        f.write("- `requirement2_final_report.md`: This report\n")
        f.write("- Various plots and visualizations in the `plots/` directory\n\n")
        
        f.write("---\n")
        f.write("*This report was generated by the AI-Powered Pharmacovigilance System*\n")
    
    print(f"✓ Final report generated: {report_file}")


def main():
    """
    Main function to run complete Requirement 2 analysis
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    print("=" * 80)
    print("REQUIREMENT 2: MODEL RISK FACTORS AND TIME-TO-EVENT ANALYSIS")
    print("AI-Powered Pharmacovigilance System")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Drugs to analyze: {args.drugs}")
    print(f"Limit per drug: {args.limit}")
    print("=" * 80)
    
    try:
        # Parse drugs list
        drugs = [drug.strip() for drug in args.drugs.split(',')]
        
        # Check if we should skip data collection
        if args.skip_data_collection or args.validation_only:
            # Try to load existing data
            combined_file = output_dir / 'data' / 'combined_survival_data.csv'
            if combined_file.exists():
                print(f"Loading existing data from: {combined_file}")
                df = pd.read_csv(combined_file)
            else:
                print("No existing data found. Please run without --skip-data-collection first.")
                sys.exit(1)
        else:
            # Phase 1: Data Collection
            df = collect_survival_data(drugs, args.limit, output_dir, args.verbose)
        
        # Phase 2: Survival Analysis
        model_results, df_with_predictions = run_survival_analysis(df, output_dir, args.verbose)
        
        # Phase 3: Validation and Testing
        validation_results = run_validation_and_testing(df_with_predictions, model_results, output_dir, args.verbose)
        
        # Phase 4: Final Report Generation
        generate_final_report(df_with_predictions, model_results, validation_results, output_dir, drugs)
        
        # Summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"✓ Total records analyzed: {len(df_with_predictions):,}")
        print(f"✓ Results saved to: {output_dir}")
        print(f"✓ Final report: {output_dir / 'reports' / 'requirement2_final_report.md'}")
        
        if 'cox_model' in model_results:
            c_index = model_results['cox_model'].get('concordance_index', 0)
            print(f"✓ Cox model concordance index: {c_index:.3f}")
        
        print("\nFiles generated:")
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(output_dir)}")
        
        print("\n" + "=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
