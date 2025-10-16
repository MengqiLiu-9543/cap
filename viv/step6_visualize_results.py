#!/usr/bin/env python3
"""
Task 5 - 步骤6: 结果可视化

生成完整的结果可视化报告
"""

import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Task 5 - 步骤 6/7: 结果可视化")
print("=" * 80)
print()

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 检查输入文件
required_files = {
    "model_comparison.csv": "模型性能比较",
    "feature_importance.csv": "特征重要性",
    "y_test.csv": "测试标签"
}

missing_files = [f for f, desc in required_files.items() if not os.path.exists(f)]

if missing_files:
    print(f"❌ 错误: 缺少必要文件:")
    for f in missing_files:
        print(f"  - {f}")
    print()
    print("请确保已运行前面的步骤")
    sys.exit(1)

print("✅ 找到所有必要文件")
print()

# 1. 模型性能比较图
print("=" * 80)
print("📊 1. 模型性能比较")
print("=" * 80)
print()

model_results = pd.read_csv("model_comparison.csv", index_col=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, y=1.00)

metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    if metric in model_results.columns:
        data = model_results[metric].sort_values()
        bars = ax.barh(range(len(data)), data.values)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.set_xlabel(title)
        ax.set_xlim([0, 1])
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, data.values)):
            ax.text(value + 0.02, i, f'{value:.3f}', va='center')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 已保存: model_performance_comparison.png")
print()

# 2. 特征重要性图
print("=" * 80)
print("📊 2. 特征重要性")
print("=" * 80)
print()

feature_imp = pd.read_csv("feature_importance.csv")

plt.figure(figsize=(10, 8))
top_n = min(15, len(feature_imp))
top_features = feature_imp.head(top_n)

plt.barh(range(top_n), top_features['importance'])
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('Importance Score')
plt.title(f'Top {top_n} Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('top_features.png', dpi=300, bbox_inches='tight')
print("✅ 已保存: top_features.png")
print()

# 3. 数据分布图
print("=" * 80)
print("📊 3. 数据分布分析")
print("=" * 80)
print()

if os.path.exists("preprocessed_data.csv"):
    df = pd.read_csv("preprocessed_data.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Distribution Analysis', fontsize=16)
    
    # 年龄分布
    if 'age_years' in df.columns or 'patientonsetage' in df.columns:
        age_col = 'age_years' if 'age_years' in df.columns else 'patientonsetage'
        age_data = pd.to_numeric(df[age_col], errors='coerce').dropna()
        if len(age_data) > 0:
            axes[0, 0].hist(age_data, bins=30, edgecolor='black')
            axes[0, 0].set_xlabel('Age (years)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Age Distribution')
    
    # 性别分布
    if 'patientsex' in df.columns:
        sex_counts = df['patientsex'].value_counts()
        sex_labels = {1: 'Male', 2: 'Female', 0: 'Unknown'}
        labels = [sex_labels.get(x, f'Code {x}') for x in sex_counts.index]
        axes[0, 1].bar(range(len(sex_counts)), sex_counts.values)
        axes[0, 1].set_xticks(range(len(sex_counts)))
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Gender Distribution')
    
    # 药物数量分布
    if 'num_drugs' in df.columns:
        drug_counts = df['num_drugs'].value_counts().sort_index()
        axes[1, 0].bar(drug_counts.index, drug_counts.values)
        axes[1, 0].set_xlabel('Number of Drugs')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Polypharmacy Distribution')
    
    # 严重程度分布
    if 'seriousnessdeath' in df.columns:
        death_counts = pd.to_numeric(df['seriousnessdeath'], errors='coerce').value_counts()
        labels = ['No Death', 'Death']
        axes[1, 1].bar(range(len(death_counts)), death_counts.values)
        axes[1, 1].set_xticks(range(len(death_counts)))
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Outcome Distribution')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: data_distribution.png")
    print()

# 生成总结报告
print("=" * 80)
print("📝 生成总结报告")
print("=" * 80)
print()

with open("RESULTS_SUMMARY.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("Task 5: Adverse Event Severity Prediction - Results Summary\n")
    f.write("=" * 80 + "\n\n")
    
    # 数据概览
    if os.path.exists("epcoritamab_data.csv"):
        df_raw = pd.read_csv("epcoritamab_data.csv")
        f.write(f"1. Data Overview\n")
        f.write(f"   Total Records: {len(df_raw)}\n")
        f.write(f"   Total Features: {len(df_raw.columns)}\n\n")
    
    # 模型性能
    f.write(f"2. Model Performance\n\n")
    f.write(model_results.to_string())
    f.write("\n\n")
    
    # 最佳模型
    best_model = model_results['accuracy'].idxmax()
    best_acc = model_results.loc[best_model, 'accuracy']
    f.write(f"3. Best Model\n")
    f.write(f"   Model: {best_model}\n")
    f.write(f"   Accuracy: {best_acc:.4f}\n\n")
    
    # Top特征
    f.write(f"4. Top 10 Most Important Features\n\n")
    f.write(feature_imp.head(10).to_string(index=False))
    f.write("\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("End of Summary\n")
    f.write("=" * 80 + "\n")

print("✅ 已保存: RESULTS_SUMMARY.txt")
print()

# 列出所有生成的文件
print("=" * 80)
print("✅ 步骤6完成 - 可视化完毕")
print("=" * 80)
print()

print("📁 生成的可视化文件:")
print("  1. model_performance_comparison.png - 模型性能对比图")
print("  2. top_features.png - 重要特征图")
print("  3. data_distribution.png - 数据分布图")
print("  4. RESULTS_SUMMARY.txt - 结果总结报告")
print()

print("💡 查看结果:")
print("  查看图片: open model_performance_comparison.png")
print("  查看报告: open RESULTS_SUMMARY.txt")
print()

print("=" * 80)
print("🎉 恭喜！Task 5 全部完成！")
print("=" * 80)
print()

print("📋 完整的输出文件清单:")
print()
print("数据文件:")
print("  ✓ epcoritamab_data.csv - 原始数据")
print("  ✓ preprocessed_data.csv - 预处理数据")
print("  ✓ X_train.csv, y_train.csv - 训练集")
print("  ✓ X_test.csv, y_test.csv - 测试集")
print()

print("模型文件:")
model_files = [f for f in os.listdir('.') if f.startswith('trained_model_')]
for f in model_files:
    print(f"  ✓ {f}")
print()

print("结果文件:")
print("  ✓ model_comparison.csv - 模型性能表")
print("  ✓ feature_importance.csv - 特征重要性表")
print()

print("可视化文件:")
print("  ✓ model_performance_comparison.png")
print("  ✓ top_features.png")
print("  ✓ data_distribution.png")
print("  ✓ feature_importance.png")
print()

print("报告文件:")
print("  ✓ RESULTS_SUMMARY.txt")
print()

print("🎯 后续工作建议:")
print("  1. 查看RESULTS_SUMMARY.txt了解整体结果")
print("  2. 检查可视化图表理解模型表现")
print("  3. 根据特征重要性优化模型")
print("  4. 准备项目报告和展示材料")
print()

