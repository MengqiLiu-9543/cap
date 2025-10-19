#!/usr/bin/env python3
"""
Task 5 - 步骤2: 检查数据

详细检查提取的数据质量
"""

import sys
import pandas as pd
import os

print("=" * 80)
print("Task 5 - 步骤 2/7: 检查数据")
print("=" * 80)
print()

# 检查数据文件（优先级：Task5专用 > 完整数据 > 多药物 > 单一药物）
DATA_FILES = ["task5_severity_prediction_data.csv", "oncology_drugs_complete.csv", 
              "oncology_drugs_data.csv", "epcoritamab_data.csv"]
DATA_FILE = None

for f in DATA_FILES:
    if os.path.exists(f):
        DATA_FILE = f
        break

if DATA_FILE is None:
    print(f"❌ 错误: 找不到数据文件")
    print()
    print("请先运行: python step1_extract_data.py")
    sys.exit(1)

print(f"✅ 找到数据文件: {DATA_FILE}")
if DATA_FILE == "task5_severity_prediction_data.csv":
    print("   (Task 5专用数据 - 35种药物)")
elif DATA_FILE == "oncology_drugs_complete.csv":
    print("   (完整肿瘤药物数据)")
elif DATA_FILE == "oncology_drugs_data.csv":
    print("   (多种肿瘤药物数据)")
else:
    print("   (单一药物数据)")
print()

# 加载数据
print("📂 加载数据...")
df = pd.read_csv(DATA_FILE)
print(f"✅ 数据加载成功")
print()

# 基本信息
print("=" * 80)
print("📊 数据基本信息")
print("=" * 80)
print()

print(f"记录数: {len(df)}")
print(f"字段数: {len(df.columns)}")
print()

print("字段列表:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# 数据类型
print("=" * 80)
print("🔍 数据质量检查")
print("=" * 80)
print()

# 缺失值
print("缺失值统计:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '字段': missing.index,
    '缺失数': missing.values,
    '缺失率': missing_pct.values
})
missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)

if len(missing_df) > 0:
    print(missing_df.head(10).to_string(index=False))
else:
    print("  ✅ 没有缺失值")
print()

# 药物分布（如果有多药物）
if 'drug_name' in df.columns:
    print("=" * 80)
    print("💊 药物分布")
    print("=" * 80)
    print()
    
    drug_counts = df['drug_name'].value_counts()
    print(f"涉及药物数: {len(drug_counts)}")
    print()
    print("Top 10 药物:")
    for drug, count in drug_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"  {drug:20s}: {count:5d} ({pct:5.1f}%)")
    print()

# 严重程度分析
print("=" * 80)
print("🏥 严重程度分析")
print("=" * 80)
print()

severity_fields = {
    'serious': '严重事件',
    'seriousnessdeath': '死亡',
    'seriousnesshospitalization': '住院',
    'seriousnesslifethreatening': '危及生命',
    'seriousnessdisabling': '致残',
    'seriousnesscongenitalanomali': '先天异常',
    'seriousnessother': '其他严重'
}

for field, label in severity_fields.items():
    if field in df.columns:
        count = pd.to_numeric(df[field], errors='coerce').fillna(0).sum()
        pct = (count / len(df)) * 100
        print(f"{label:12s}: {int(count):4d} 例 ({pct:5.1f}%)")

print()

# 患者人口统计
print("=" * 80)
print("👥 患者人口统计")
print("=" * 80)
print()

# 性别分布
if 'patientsex' in df.columns:
    print("性别分布:")
    sex_map = {1: '男性', 2: '女性', 0: '未知'}
    sex_counts = df['patientsex'].value_counts()
    for sex_code, count in sex_counts.items():
        label = sex_map.get(sex_code, f'代码{sex_code}')
        pct = (count / len(df)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

# 年龄统计
if 'patientonsetage' in df.columns:
    age_data = pd.to_numeric(df['patientonsetage'], errors='coerce').dropna()
    if len(age_data) > 0:
        print("年龄统计:")
        print(f"  样本数: {len(age_data)}")
        print(f"  平均年龄: {age_data.mean():.1f}")
        print(f"  中位数: {age_data.median():.1f}")
        print(f"  最小值: {age_data.min():.1f}")
        print(f"  最大值: {age_data.max():.1f}")
        print()

# 药物使用
if 'num_drugs' in df.columns:
    print("药物使用统计:")
    print(f"  平均药物数: {df['num_drugs'].mean():.1f}")
    print(f"  最多药物数: {df['num_drugs'].max()}")
    polypharmacy = (df['num_drugs'] > 1).sum()
    print(f"  多药使用: {polypharmacy} ({polypharmacy/len(df)*100:.1f}%)")
    print()

# 数据样本
print("=" * 80)
print("📋 数据样本")
print("=" * 80)
print()
print(df.head(5))
print()

# 总结
print("=" * 80)
print("✅ 步骤2完成 - 数据检查完毕")
print("=" * 80)
print()

print("✅ 数据质量评估:")
if len(df) >= 1000:
    print("  🌟 优秀: 数据量充足，适合建模")
elif len(df) >= 500:
    print("  ✅ 良好: 数据量足够，可以建模")
else:
    print("  ⚠️  数据量较少，但仍可进行初步分析")
print()

print("🎯 下一步:")
print("  运行: python step3_preprocess_data.py")
print("  作用: 数据预处理和特征工程")
print()

