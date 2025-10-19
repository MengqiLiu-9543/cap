#!/usr/bin/env python3
"""
Task 5 - 步骤5: 特征重要性分析

分析哪些特征对预测最重要
"""

import sys
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

print("=" * 80)
print("Task 5 - 步骤 5/7: 特征重要性分析")
print("=" * 80)
print()

# 检查输入文件
if not os.path.exists("X_train.csv"):
    print("❌ 错误: 找不到 X_train.csv")
    print("请先运行: python step3_preprocess_data.py")
    sys.exit(1)

# 查找训练好的模型
model_files = [f for f in os.listdir('.') if f.startswith('trained_model_') and f.endswith('.pkl')]

if not model_files:
    print("❌ 错误: 找不到训练好的模型")
    print("请先运行: python step4_train_models.py")
    sys.exit(1)

print(f"✅ 找到 {len(model_files)} 个训练好的模型")
print()

# 加载特征名称
X_train = pd.read_csv("X_train.csv")
feature_names = X_train.columns.tolist()
print(f"特征数量: {len(feature_names)}")
print()

# 选择最佳模型进行分析
print("=" * 80)
print("🔍 选择模型进行分析")
print("=" * 80)
print()

# 优先选择树模型（有feature_importances_属性）
priority_models = ['random_forest', 'gradient_boosting', 'xgboost']
selected_model = None
selected_file = None

for model_name in priority_models:
    for f in model_files:
        if model_name in f:
            selected_model = model_name
            selected_file = f
            break
    if selected_model:
        break

if not selected_model:
    # 如果没有树模型，使用第一个
    selected_file = model_files[0]
    selected_model = selected_file.replace('trained_model_', '').replace('.pkl', '')

print(f"分析模型: {selected_model}")
print(f"模型文件: {selected_file}")
print()

# 加载模型
print("📂 加载模型...")
try:
    with open(selected_file, 'rb') as f:
        model = pickle.load(f)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    sys.exit(1)

print()

# 提取特征重要性
print("=" * 80)
print("📊 特征重要性分析")
print("=" * 80)
print()

try:
    if hasattr(model, 'feature_importances_'):
        # 树模型
        importances = model.feature_importances_
        print("✅ 使用模型内置特征重要性")
    elif hasattr(model, 'coef_'):
        # 线性模型
        importances = abs(model.coef_[0])
        print("✅ 使用模型系数绝对值作为重要性")
    else:
        print("⚠️  该模型不支持直接提取特征重要性")
        print("   建议使用随机森林或梯度提升模型")
        sys.exit(0)
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print()
    print("Top 20 最重要特征:")
    print()
    print(feature_importance_df.head(20).to_string(index=False))
    print()
    
    # 保存结果
    print("💾 保存特征重要性...")
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    print("✅ 已保存: feature_importance.csv")
    print()
    
    # 可视化
    print("📊 生成可视化图表...")
    
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(top_n), top_features['importance'])
    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances ({selected_model})')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: feature_importance.png")
    print()
    
except Exception as e:
    print(f"❌ 分析失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 特征解释
print("=" * 80)
print("💡 特征重要性解读")
print("=" * 80)
print()

print("重要性高的特征对模型预测影响更大:")
print()

top_5 = feature_importance_df.head(5)
for i, row in enumerate(top_5.itertuples(), 1):
    print(f"{i}. {row.feature}")
    print(f"   重要性: {row.importance:.4f}")
    
    # 特征说明
    if 'age' in row.feature.lower():
        print(f"   说明: 患者年龄相关特征")
    elif 'drug' in row.feature.lower():
        print(f"   说明: 药物使用相关特征")
    elif 'sex' in row.feature.lower():
        print(f"   说明: 患者性别")
    elif 'reaction' in row.feature.lower():
        print(f"   说明: 不良反应数量")
    elif 'polypharmacy' in row.feature.lower():
        print(f"   说明: 是否同时使用多种药物")
    print()

# 总结
print("=" * 80)
print("✅ 步骤5完成 - 特征分析完毕")
print("=" * 80)
print()

print("📁 生成的文件:")
print("  1. feature_importance.csv - 特征重要性表格")
print("  2. feature_importance.png - 特征重要性图表")
print()

print("🎯 下一步:")
print("  运行: python step6_visualize_results.py")
print("  作用: 生成完整的结果可视化")
print()

print("💡 提示:")
print("  查看图表: open feature_importance.png")
print()

