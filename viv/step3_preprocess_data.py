#!/usr/bin/env python3
"""
Task 5 - 步骤 3/7: 数据预处理（简化版）

针对extract_task5_data.py提取的数据进行预处理
"""

import sys
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

print("=" * 80)
print("Task 5 - 步骤 3/7: 数据预处理")
print("=" * 80)
print()

# 检查输入文件
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
    print("请先运行: python extract_task5_data.py")
    sys.exit(1)

print(f"✅ 找到输入文件: {DATA_FILE}")
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
print("📂 加载原始数据...")
df = pd.read_csv(DATA_FILE)
print(f"✅ 原始数据: {len(df)} 行, {len(df.columns)} 列")
print()

# 数据预处理
print("=" * 80)
print("🔄 执行数据预处理")
print("=" * 80)
print()

print("处理步骤:")
print("  1️⃣  转换数值字段")
print("  2️⃣  特征工程（创建新特征）")
print("  3️⃣  处理缺失值")
print("  4️⃣  清理数据")
print()

# 步骤1: 转换数值字段
print("1️⃣  转换数值字段...")

# 转换严重性指标为数值
severity_cols = ['serious', 'seriousnessdeath', 'seriousnesshospitalization',
                 'seriousnesslifethreatening', 'seriousnessdisabling',
                 'seriousnesscongenitalanomali', 'seriousnessother']

for col in severity_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # 转换为二分类：任何非0值都视为1
        df[col] = (df[col] > 0).astype(int)

# 转换患者信息
if 'patientsex' in df.columns:
    df['patientsex'] = pd.to_numeric(df['patientsex'], errors='coerce').fillna(0).astype(int)

if 'patientonsetage' in df.columns:
    df['patientonsetage'] = pd.to_numeric(df['patientonsetage'], errors='coerce')

if 'patientweight' in df.columns:
    df['patientweight'] = pd.to_numeric(df['patientweight'], errors='coerce')

# 转换药物和反应数量
if 'num_drugs' in df.columns:
    df['num_drugs'] = pd.to_numeric(df['num_drugs'], errors='coerce').fillna(1).astype(int)

if 'num_reactions' in df.columns:
    df['num_reactions'] = pd.to_numeric(df['num_reactions'], errors='coerce').fillna(1).astype(int)

print("✅ 数值转换完成")
print()

# 步骤2: 特征工程
print("2️⃣  特征工程...")

# 创建年龄特征（处理异常值）
if 'patientonsetage' in df.columns:
    # 将异常的年龄值设为缺失（年龄应该在0-120之间）
    df.loc[df['patientonsetage'] > 120, 'patientonsetage'] = np.nan
    df.loc[df['patientonsetage'] < 0, 'patientonsetage'] = np.nan
    
    # 年龄分组
    df['age_group'] = pd.cut(df['patientonsetage'], 
                              bins=[0, 18, 45, 65, 120], 
                              labels=['0-18', '19-45', '46-65', '66+'],
                              include_lowest=True)
    
    # 年龄分组 one-hot encoding
    age_dummies = pd.get_dummies(df['age_group'], prefix='age')
    df = pd.concat([df, age_dummies], axis=1)
    
    # 创建年龄缺失标志
    df['age_missing'] = df['patientonsetage'].isna().astype(int)

# 创建性别特征
if 'patientsex' in df.columns:
    df['sex_male'] = (df['patientsex'] == 1).astype(int)
    df['sex_female'] = (df['patientsex'] == 2).astype(int)
    df['sex_unknown'] = (df['patientsex'] == 0).astype(int)

# 创建多药使用特征
if 'num_drugs' in df.columns:
    df['polypharmacy'] = (df['num_drugs'] > 1).astype(int)
    df['high_polypharmacy'] = (df['num_drugs'] > 5).astype(int)

# 创建多反应特征
if 'num_reactions' in df.columns:
    df['multiple_reactions'] = (df['num_reactions'] > 1).astype(int)
    df['many_reactions'] = (df['num_reactions'] > 3).astype(int)

# 创建综合严重性评分（排除目标变量 seriousnessdeath 避免数据泄露）
severity_score = 0
# 注意：不包含 seriousnessdeath，因为它是我们的目标变量
if 'seriousnesslifethreatening' in df.columns:
    severity_score += df['seriousnesslifethreatening'] * 4
if 'seriousnesshospitalization' in df.columns:
    severity_score += df['seriousnesshospitalization'] * 3
if 'seriousnessdisabling' in df.columns:
    severity_score += df['seriousnessdisabling'] * 3
if 'seriousnessother' in df.columns:
    severity_score += df['seriousnessother'] * 1
df['severity_score'] = severity_score

print("✅ 特征工程完成")
print(f"   新增特征数: {len(df.columns) - len(pd.read_csv(DATA_FILE).columns)}")
print()

# 保存预处理后的数据
print("💾 保存预处理数据...")
df.to_csv("preprocessed_data.csv", index=False)
print("✅ 已保存: preprocessed_data.csv")
print()

# 显示新增特征
print("🆕 新增特征列表:")
original_cols = set(pd.read_csv(DATA_FILE).columns)
new_cols = [col for col in df.columns if col not in original_cols]
for i, col in enumerate(new_cols, 1):
    print(f"  {i:2d}. {col}")
print()

# 步骤3: 准备训练数据
print("=" * 80)
print("🎯 准备训练数据")
print("=" * 80)
print()

print("目标变量: seriousnessdeath (预测是否导致死亡)")
print()

# 创建目标变量
y = df['seriousnessdeath'].copy()
positive_count = y.sum()
negative_count = len(y) - positive_count

print(f"目标变量分布:")
print(f"  阳性样本 (死亡): {positive_count} ({positive_count/len(y)*100:.1f}%)")
print(f"  阴性样本 (存活): {negative_count} ({negative_count/len(y)*100:.1f}%)")
print()

# 选择特征
print("🔍 选择特征...")

# 排除不能用于预测的字段
exclude_cols = [
    'safetyreportid',  # ID
    'receivedate',  # 日期
    'target_drug',  # 药物名称（文本）
    'drugname',  # 药物名称（文本）
    'all_drugs',  # 药物列表（文本）
    'drug_indication',  # 指征（文本）
    'reactions',  # 反应列表（文本）
    'patientonsetageunit',  # 年龄单位（文本）
    'age_group',  # 分类变量（已转为dummy）
    'reporter_qualification',  # 报告者资质（文本）
    # 目标变量及相关严重性指标
    'seriousnessdeath',
    'serious',
    'seriousnesshospitalization',
    'seriousnesslifethreatening',
    'seriousnessdisabling',
    'seriousnesscongenitalanomali',
    'seriousnessother'
]

# 选择数值特征
feature_cols = [col for col in df.columns 
                if col not in exclude_cols and 
                df[col].dtype in ['int64', 'float64', 'uint8']]

X = df[feature_cols].copy()

print(f"✅ 选择了 {len(feature_cols)} 个特征")
print()

print("特征列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")
print()

# 步骤4: 处理缺失值
print("🔧 处理特征缺失值...")
print(f"   缺失值统计:")
missing_counts = X.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        pct = count / len(X) * 100
        print(f"     {col}: {count} ({pct:.1f}%)")
else:
    print("     无缺失值")

# 使用中位数填充缺失值
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)
print("✅ 缺失值处理完成")
print()

# 步骤5: 划分训练集和测试集
print("📊 划分训练集和测试集...")

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # 保持类别比例
)

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")
print()

print(f"训练集目标分布:")
print(f"  阳性: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"  阴性: {len(y_train)-y_train.sum()} ({(len(y_train)-y_train.sum())/len(y_train)*100:.1f}%)")
print()

# 保存数据集
print("💾 保存训练和测试数据...")
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False, header=['seriousnessdeath'])
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False, header=['seriousnessdeath'])

print("✅ 已保存:")
print("  - X_train.csv")
print("  - y_train.csv")
print("  - X_test.csv")
print("  - y_test.csv")
print()

# 数据样本
print("=" * 80)
print("📋 预处理后数据样本")
print("=" * 80)
print()
print(df[['target_drug', 'seriousnessdeath', 'patientonsetage', 'patientsex', 
          'num_drugs', 'num_reactions']].head(3))
print()

# 总结
print("=" * 80)
print("✅ 步骤3完成 - 数据预处理完毕")
print("=" * 80)
print()

print("📁 生成的文件:")
print("  1. preprocessed_data.csv - 完整预处理数据")
print("  2. X_train.csv - 训练特征")
print("  3. y_train.csv - 训练标签")
print("  4. X_test.csv - 测试特征")
print("  5. y_test.csv - 测试标签")
print()

print("🎯 下一步:")
print("  运行: python step4_train_models.py")
print("  作用: 训练多个机器学习模型")
print()

