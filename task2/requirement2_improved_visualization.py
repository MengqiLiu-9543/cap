#!/usr/bin/env python3
"""
Requirement 2: 改进的可视化脚本
修复图表标签显示问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_improved_visualizations(df, feature_importance, drug_patterns, output_dir='./'):
    """
    创建改进的可视化图表，修复标签显示问题
    """
    print("\n创建改进的可视化图表...")
    
    # 设置中文字体和更大的图表
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    
    # 创建更大的图表
    fig = plt.figure(figsize=(24, 16))
    
    # 1. 特征重要性 - 改进标签
    ax1 = plt.subplot(3, 3, 1)
    top_features = feature_importance.head(10)
    
    # 简化特征名称
    feature_names = []
    for feature in top_features['feature']:
        if 'patient_age' in feature:
            feature_names.append('患者年龄')
        elif 'total_events' in feature:
            feature_names.append('事件总数')
        elif 'time_to_event_days' in feature:
            feature_names.append('时间到事件')
        elif 'total_drugs' in feature:
            feature_names.append('药物总数')
        elif 'concomitant_drugs' in feature:
            feature_names.append('伴随药物')
        elif 'administration_route' in feature:
            feature_names.append('给药途径')
        elif 'patient_weight' in feature:
            feature_names.append('患者体重')
        elif 'drug_characterization' in feature:
            feature_names.append('药物特征')
        elif 'weight_group' in feature:
            feature_names.append('体重组')
        elif 'polypharmacy' in feature:
            feature_names.append('多药联用')
        else:
            feature_names.append(feature)
    
    bars1 = ax1.barh(range(len(top_features)), top_features['importance'])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(feature_names, fontsize=10)
    ax1.set_title('Top 10 风险因素', fontsize=12, fontweight='bold')
    ax1.set_xlabel('重要性', fontsize=10)
    
    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars1, top_features['importance'])):
        ax1.text(importance + 0.001, i, f'{importance:.3f}', 
                va='center', fontsize=9)
    
    # 2. 年龄组分布 - 改进
    ax2 = plt.subplot(3, 3, 2)
    age_counts = df['age_group'].value_counts()
    bars2 = ax2.bar(age_counts.index, age_counts.values, color='skyblue')
    ax2.set_title('年龄组分布', fontsize=12, fontweight='bold')
    ax2.set_ylabel('记录数', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # 添加数值标签
    for bar, count in zip(bars2, age_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    # 3. 严重事件率 by 年龄组 - 改进
    ax3 = plt.subplot(3, 3, 3)
    age_serious = df.groupby('age_group')['is_serious'].mean()
    bars3 = ax3.bar(age_serious.index, age_serious.values, color='lightcoral')
    ax3.set_title('严重事件率（按年龄组）', fontsize=12, fontweight='bold')
    ax3.set_ylabel('严重事件率', fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    
    # 添加数值标签
    for bar, rate in zip(bars3, age_serious.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 时间到事件分布 - 改进
    ax4 = plt.subplot(3, 3, 4)
    time_data = df[df['time_to_event_days'] > 0]['time_to_event_days']
    ax4.hist(time_data, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    ax4.set_title('时间到事件分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('天数', fontsize=10)
    ax4.set_ylabel('频率', fontsize=10)
    ax4.tick_params(labelsize=10)
    
    # 添加统计信息
    median_time = time_data.median()
    ax4.axvline(median_time, color='red', linestyle='--', linewidth=2)
    ax4.text(median_time, ax4.get_ylim()[1]*0.8, f'中位数: {median_time:.1f}天', 
            fontsize=9, ha='center')
    
    # 5. 长期 vs 其他事件时间对比 - 改进
    ax5 = plt.subplot(3, 3, 5)
    lt_times = df[df['is_long_term_event'] == 1]['time_to_event_days']
    other_times = df[df['is_long_term_event'] == 0]['time_to_event_days']
    
    ax5.hist([lt_times[lt_times > 0], other_times[other_times > 0]], 
             bins=30, alpha=0.7, label=['长期事件', '其他事件'],
             color=['orange', 'lightblue'])
    ax5.set_title('时间对比：长期 vs 其他事件', fontsize=12, fontweight='bold')
    ax5.set_xlabel('天数', fontsize=10)
    ax5.set_ylabel('频率', fontsize=10)
    ax5.legend(fontsize=10)
    ax5.tick_params(labelsize=10)
    
    # 6. Top 10 药物严重事件率 - 改进
    ax6 = plt.subplot(3, 3, 6)
    top_drugs = drug_patterns.head(10)
    
    # 简化药物名称
    drug_names = []
    for drug in top_drugs.index:
        if len(drug) > 12:
            drug_names.append(drug[:10] + '...')
        else:
            drug_names.append(drug)
    
    bars6 = ax6.barh(range(len(top_drugs)), top_drugs['serious_rate'], color='salmon')
    ax6.set_yticks(range(len(top_drugs)))
    ax6.set_yticklabels(drug_names, fontsize=9)
    ax6.set_title('Top 10 药物严重事件率', fontsize=12, fontweight='bold')
    ax6.set_xlabel('严重事件率', fontsize=10)
    ax6.tick_params(labelsize=10)
    
    # 添加数值标签
    for i, (bar, rate) in enumerate(zip(bars6, top_drugs['serious_rate'])):
        ax6.text(rate + 0.01, i, f'{rate:.3f}', 
                va='center', fontsize=8)
    
    # 7. 感染 vs 恶性肿瘤率对比 - 改进
    ax7 = plt.subplot(3, 3, 7)
    event_types = ['感染', '继发恶性肿瘤', '其他']
    event_counts = [
        df['is_infection'].sum(),
        df['is_secondary_malignancy'].sum(),
        len(df) - df['is_infection'].sum() - df['is_secondary_malignancy'].sum()
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    wedges, texts, autotexts = ax7.pie(event_counts, labels=event_types, 
                                       autopct='%1.1f%%', colors=colors,
                                       textprops={'fontsize': 10})
    ax7.set_title('事件类型分布', fontsize=12, fontweight='bold')
    
    # 8. 多药联用风险 - 改进
    ax8 = plt.subplot(3, 3, 8)
    poly_serious = df.groupby('polypharmacy')['is_serious'].mean()
    bars8 = ax8.bar(['无多药联用', '多药联用'], poly_serious.values, 
                    color=['lightblue', 'lightcoral'])
    ax8.set_title('多药联用与严重事件率', fontsize=12, fontweight='bold')
    ax8.set_ylabel('严重事件率', fontsize=10)
    ax8.tick_params(labelsize=10)
    
    # 添加数值标签
    for bar, rate in zip(bars8, poly_serious.values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 9. Top 不良事件 - 改进
    ax9 = plt.subplot(3, 3, 9)
    top_ae = df['adverse_event'].value_counts().head(15)
    
    # 简化不良事件名称
    ae_names = []
    for ae in top_ae.index:
        if len(ae) > 20:
            ae_names.append(ae[:17] + '...')
        else:
            ae_names.append(ae)
    
    bars9 = ax9.barh(range(len(top_ae)), top_ae.values, color='lightsteelblue')
    ax9.set_yticks(range(len(top_ae)))
    ax9.set_yticklabels(ae_names, fontsize=8)
    ax9.set_title('Top 15 不良事件', fontsize=12, fontweight='bold')
    ax9.set_xlabel('发生次数', fontsize=10)
    ax9.tick_params(labelsize=9)
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars9, top_ae.values)):
        ax9.text(count + max(top_ae.values)*0.01, i, str(count), 
                va='center', fontsize=8)
    
    # 调整布局，增加间距
    plt.tight_layout(pad=3.0)
    
    # 保存高质量图片
    plt.savefig(f'{output_dir}/requirement2_improved_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 改进的可视化已保存: {output_dir}/requirement2_improved_analysis.png")

def main():
    """
    主函数 - 重新生成改进的可视化
    """
    print("="*80)
    print("Requirement 2: 改进可视化生成")
    print("="*80)
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('requirement2_faers_data.csv')
    feature_importance = pd.read_csv('requirement2_feature_importance_final.csv')
    drug_patterns = pd.read_csv('requirement2_drug_safety_profiles.csv', index_col=0)
    
    print(f"数据加载完成: {len(df)} 条记录")
    
    # 创建改进的可视化
    create_improved_visualizations(df, feature_importance, drug_patterns)
    
    print("\n" + "="*80)
    print("改进的可视化生成完成！")
    print("="*80)
    print("新文件: requirement2_improved_analysis.png")
    print("改进内容:")
    print("  ✓ 更大的字体和图表")
    print("  ✓ 中文标签")
    print("  ✓ 数值标签显示")
    print("  ✓ 更好的颜色搭配")
    print("  ✓ 简化的长标签")
    print("  ✓ 增加图表间距")
    print("="*80)

if __name__ == "__main__":
    main()
