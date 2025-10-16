#!/usr/bin/env python3
"""
Requirement 2: 完整分析 - 使用真实FAERS数据
运行完整的生存分析和风险因素建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filename='requirement2_faers_data.csv'):
    """
    加载并准备数据
    """
    print("加载数据...")
    df = pd.read_csv(filename)
    
    print(f"原始数据: {len(df)} 条记录")
    
    # 数据清洗
    df = df.dropna(subset=['adverse_event'])
    df = df[df['time_to_event_days'] >= 0]
    df = df[df['time_to_event_days'] <= 3650]  # 限制在10年内
    
    # 编码分类变量
    categorical_cols = ['age_group', 'weight_group', 'drug_interaction_risk', 'administration_route']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    # 处理数值列
    numeric_cols = ['patient_age', 'patient_weight', 'total_drugs', 'total_events', 
                   'concomitant_drugs', 'time_to_event_days']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 处理布尔列
    boolean_cols = ['polypharmacy', 'multiple_events']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # 处理drug_characterization
    if 'drug_characterization' in df.columns:
        df['drug_characterization'] = pd.to_numeric(df['drug_characterization'], errors='coerce').fillna(0)
    
    print(f"清洗后数据: {len(df)} 条记录")
    print(f"药物数: {df['target_drug'].nunique()}")
    print(f"不良事件数: {df['adverse_event'].nunique()}")
    
    return df

def perform_risk_analysis(df):
    """
    执行风险因素分析
    """
    print("\n" + "="*80)
    print("风险因素分析")
    print("="*80)
    
    # 准备特征
    feature_cols = [
        'age_group_encoded', 'weight_group_encoded', 'drug_interaction_risk_encoded',
        'administration_route_encoded', 'patient_age', 'patient_weight', 'total_drugs',
        'total_events', 'concomitant_drugs', 'polypharmacy', 'multiple_events',
        'drug_characterization', 'time_to_event_days'
    ]
    
    # 过滤可用特征
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['is_serious']
    
    print(f"\n使用特征: {len(available_features)}")
    print(f"数据集大小: {X.shape}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练随机森林模型
    print("\n训练随机森林模型...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # 评估
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    
    print(f"\n模型性能:")
    print(f"  AUC Score: {auc_score:.3f}")
    print(f"  Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 重要特征:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return rf, feature_importance, auc_score

def analyze_long_term_events(df):
    """
    分析长期不良事件
    """
    print("\n" + "="*80)
    print("长期不良事件分析")
    print("="*80)
    
    # 长期事件统计
    long_term_df = df[df['is_long_term_event'] == 1]
    infection_df = df[df['is_infection'] == 1]
    malignancy_df = df[df['is_secondary_malignancy'] == 1]
    
    print(f"\n总体统计:")
    print(f"  长期事件率: {len(long_term_df)/len(df):.3f}")
    print(f"  感染率: {len(infection_df)/len(df):.3f}")
    print(f"  继发恶性肿瘤率: {len(malignancy_df)/len(df):.3f}")
    
    if len(long_term_df) > 0:
        print(f"\n时间统计:")
        print(f"  长期事件中位时间: {long_term_df['time_to_event_days'].median():.1f} 天")
        
        if len(infection_df) > 0:
            print(f"  感染中位时间: {infection_df['time_to_event_days'].median():.1f} 天")
            print(f"  感染严重事件率: {infection_df['is_serious'].mean():.3f}")
        
        if len(malignancy_df) > 0:
            print(f"  继发恶性肿瘤中位时间: {malignancy_df['time_to_event_days'].median():.1f} 天")
            print(f"  继发恶性肿瘤严重事件率: {malignancy_df['is_serious'].mean():.3f}")
    
    # 按年龄组分析
    print(f"\n按年龄组分析:")
    age_patterns = df.groupby('age_group').agg({
        'is_serious': 'mean',
        'is_long_term_event': 'mean',
        'is_infection': 'mean',
        'is_secondary_malignancy': 'mean',
        'time_to_event_days': 'median'
    }).round(3)
    
    for age_group, row in age_patterns.iterrows():
        print(f"  {age_group}:")
        print(f"    严重事件率: {row['is_serious']:.3f}")
        print(f"    长期事件率: {row['is_long_term_event']:.3f}")
        print(f"    感染率: {row['is_infection']:.3f}")
        print(f"    中位时间: {row['time_to_event_days']:.1f} 天")
    
    return long_term_df, infection_df, malignancy_df

def analyze_drug_profiles(df):
    """
    分析药物特异性安全概况
    """
    print("\n" + "="*80)
    print("药物特异性安全概况")
    print("="*80)
    
    drug_patterns = df.groupby('target_drug').agg({
        'is_serious': ['count', 'mean'],
        'is_long_term_event': 'mean',
        'is_infection': 'mean',
        'is_secondary_malignancy': 'mean',
        'time_to_event_days': 'median'
    }).round(3)
    
    drug_patterns.columns = ['count', 'serious_rate', 'longterm_rate', 'infection_rate', 
                              'malignancy_rate', 'median_time']
    drug_patterns = drug_patterns.sort_values('serious_rate', ascending=False)
    
    print(f"\nTop 10 严重事件率最高的药物:")
    print(drug_patterns.head(10).to_string())
    
    print(f"\nTop 10 长期事件率最高的药物:")
    print(drug_patterns.sort_values('longterm_rate', ascending=False).head(10)[['longterm_rate', 'infection_rate', 'malignancy_rate']].to_string())
    
    return drug_patterns

def create_visualizations(df, feature_importance, drug_patterns, output_dir='./'):
    """
    创建可视化图表
    """
    print("\n创建可视化图表...")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 特征重要性
    ax1 = plt.subplot(3, 3, 1)
    top_features = feature_importance.head(10)
    ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_title('Top 10 风险因素')
    ax1.set_xlabel('重要性')
    
    # 2. 年龄组分布
    ax2 = plt.subplot(3, 3, 2)
    age_counts = df['age_group'].value_counts()
    ax2.bar(age_counts.index, age_counts.values)
    ax2.set_title('年龄组分布')
    ax2.set_ylabel('记录数')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 严重事件率 by 年龄组
    ax3 = plt.subplot(3, 3, 3)
    age_serious = df.groupby('age_group')['is_serious'].mean()
    ax3.bar(age_serious.index, age_serious.values)
    ax3.set_title('严重事件率（按年龄组）')
    ax3.set_ylabel('严重事件率')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 时间到事件分布
    ax4 = plt.subplot(3, 3, 4)
    time_data = df[df['time_to_event_days'] > 0]['time_to_event_days']
    ax4.hist(time_data, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_title('时间到事件分布')
    ax4.set_xlabel('天数')
    ax4.set_ylabel('频率')
    
    # 5. 长期 vs 其他事件时间对比
    ax5 = plt.subplot(3, 3, 5)
    lt_times = df[df['is_long_term_event'] == 1]['time_to_event_days']
    other_times = df[df['is_long_term_event'] == 0]['time_to_event_days']
    ax5.hist([lt_times[lt_times > 0], other_times[other_times > 0]], 
             bins=30, alpha=0.7, label=['长期事件', '其他事件'])
    ax5.set_title('时间对比：长期 vs 其他事件')
    ax5.set_xlabel('天数')
    ax5.set_ylabel('频率')
    ax5.legend()
    
    # 6. Top 10 药物严重事件率
    ax6 = plt.subplot(3, 3, 6)
    top_drugs = drug_patterns.head(10)
    ax6.barh(range(len(top_drugs)), top_drugs['serious_rate'])
    ax6.set_yticks(range(len(top_drugs)))
    ax6.set_yticklabels(top_drugs.index)
    ax6.set_title('Top 10 药物严重事件率')
    ax6.set_xlabel('严重事件率')
    
    # 7. 感染 vs 恶性肿瘤率对比
    ax7 = plt.subplot(3, 3, 7)
    event_types = ['感染', '继发恶性肿瘤', '其他']
    event_counts = [
        df['is_infection'].sum(),
        df['is_secondary_malignancy'].sum(),
        len(df) - df['is_infection'].sum() - df['is_secondary_malignancy'].sum()
    ]
    ax7.pie(event_counts, labels=event_types, autopct='%1.1f%%')
    ax7.set_title('事件类型分布')
    
    # 8. 多药联用风险
    ax8 = plt.subplot(3, 3, 8)
    poly_serious = df.groupby('polypharmacy')['is_serious'].mean()
    ax8.bar(['无多药联用', '多药联用'], poly_serious.values)
    ax8.set_title('多药联用与严重事件率')
    ax8.set_ylabel('严重事件率')
    
    # 9. Top 不良事件
    ax9 = plt.subplot(3, 3, 9)
    top_ae = df['adverse_event'].value_counts().head(15)
    ax9.barh(range(len(top_ae)), top_ae.values)
    ax9.set_yticks(range(len(top_ae)))
    ax9.set_yticklabels(top_ae.index, fontsize=8)
    ax9.set_title('Top 15 不良事件')
    ax9.set_xlabel('发生次数')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/requirement2_full_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化已保存: {output_dir}/requirement2_full_analysis.png")

def generate_report(df, rf_model, feature_importance, auc_score, drug_patterns, output_file='requirement2_final_report.txt'):
    """
    生成最终报告
    """
    print("\n生成最终报告...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REQUIREMENT 2: 风险因素和时间到事件分析 - 最终报告\n")
        f.write("AI-Powered Pharmacovigilance System\n")
        f.write("="*80 + "\n\n")
        
        f.write("数据概览\n")
        f.write("-"*50 + "\n")
        f.write(f"总记录数: {len(df):,}\n")
        f.write(f"药物数量: {df['target_drug'].nunique()}\n")
        f.write(f"不良事件类型: {df['adverse_event'].nunique():,}\n")
        f.write(f"严重事件数: {df['is_serious'].sum():,} ({df['is_serious'].mean()*100:.1f}%)\n")
        f.write(f"长期事件数: {df['is_long_term_event'].sum():,} ({df['is_long_term_event'].mean()*100:.1f}%)\n")
        f.write(f"感染事件数: {df['is_infection'].sum():,}\n")
        f.write(f"继发恶性肿瘤数: {df['is_secondary_malignancy'].sum():,}\n\n")
        
        f.write("模型性能\n")
        f.write("-"*50 + "\n")
        f.write(f"AUC Score: {auc_score:.3f}\n")
        f.write(f"模型类型: Random Forest Classifier\n")
        f.write(f"特征数量: {len(feature_importance)}\n\n")
        
        f.write("Top 15 风险因素\n")
        f.write("-"*50 + "\n")
        for i, row in feature_importance.head(15).iterrows():
            f.write(f"{i+1}. {row['feature']}: {row['importance']:.3f}\n")
        f.write("\n")
        
        f.write("长期不良事件分析\n")
        f.write("-"*50 + "\n")
        lt_df = df[df['is_long_term_event'] == 1]
        inf_df = df[df['is_infection'] == 1]
        mal_df = df[df['is_secondary_malignancy'] == 1]
        
        f.write(f"长期事件率: {len(lt_df)/len(df):.3f}\n")
        if len(lt_df) > 0:
            f.write(f"长期事件中位时间: {lt_df['time_to_event_days'].median():.1f} 天\n")
        
        if len(inf_df) > 0:
            f.write(f"感染率: {len(inf_df)/len(df):.3f}\n")
            f.write(f"感染中位时间: {inf_df['time_to_event_days'].median():.1f} 天\n")
            f.write(f"感染严重事件率: {inf_df['is_serious'].mean():.3f}\n")
        
        if len(mal_df) > 0:
            f.write(f"继发恶性肿瘤率: {len(mal_df)/len(df):.3f}\n")
            f.write(f"继发恶性肿瘤中位时间: {mal_df['time_to_event_days'].median():.1f} 天\n")
            f.write(f"继发恶性肿瘤严重事件率: {mal_df['is_serious'].mean():.3f}\n")
        f.write("\n")
        
        f.write("按年龄组风险模式\n")
        f.write("-"*50 + "\n")
        age_patterns = df.groupby('age_group').agg({
            'is_serious': 'mean',
            'is_long_term_event': 'mean',
            'time_to_event_days': 'median'
        }).round(3)
        
        for age_group, row in age_patterns.iterrows():
            f.write(f"{age_group}:\n")
            f.write(f"  严重事件率: {row['is_serious']:.3f}\n")
            f.write(f"  长期事件率: {row['is_long_term_event']:.3f}\n")
            f.write(f"  中位时间: {row['time_to_event_days']:.1f} 天\n")
        f.write("\n")
        
        f.write("Top 10 药物安全概况\n")
        f.write("-"*50 + "\n")
        for drug, row in drug_patterns.head(10).iterrows():
            f.write(f"{drug}:\n")
            f.write(f"  记录数: {int(row['count'])}\n")
            f.write(f"  严重事件率: {row['serious_rate']:.3f}\n")
            f.write(f"  长期事件率: {row['longterm_rate']:.3f}\n")
            f.write(f"  感染率: {row['infection_rate']:.3f}\n")
            f.write(f"  继发恶性肿瘤率: {row['malignancy_rate']:.3f}\n")
            f.write(f"  中位时间: {row['median_time']:.1f} 天\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"✓ 报告已保存: {output_file}")

def main():
    """
    主函数
    """
    print("="*80)
    print("REQUIREMENT 2: 完整分析 - 使用真实FAERS数据")
    print("="*80)
    
    # 1. 加载数据
    df = load_and_prepare_data()
    
    # 2. 风险因素分析
    rf_model, feature_importance, auc_score = perform_risk_analysis(df)
    
    # 3. 长期事件分析
    lt_df, inf_df, mal_df = analyze_long_term_events(df)
    
    # 4. 药物特异性分析
    drug_patterns = analyze_drug_profiles(df)
    
    # 5. 创建可视化
    create_visualizations(df, feature_importance, drug_patterns)
    
    # 6. 生成报告
    generate_report(df, rf_model, feature_importance, auc_score, drug_patterns)
    
    # 7. 保存处理后的数据
    df.to_csv('requirement2_analyzed_data.csv', index=False)
    feature_importance.to_csv('requirement2_feature_importance_final.csv', index=False)
    drug_patterns.to_csv('requirement2_drug_safety_profiles.csv')
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("生成的文件:")
    print("  - requirement2_faers_data.csv (原始数据)")
    print("  - requirement2_analyzed_data.csv (分析后数据)")
    print("  - requirement2_feature_importance_final.csv (特征重要性)")
    print("  - requirement2_drug_safety_profiles.csv (药物安全概况)")
    print("  - requirement2_final_report.txt (最终报告)")
    print("  - requirement2_full_analysis.png (可视化图表)")
    print("="*80)

if __name__ == "__main__":
    main()

