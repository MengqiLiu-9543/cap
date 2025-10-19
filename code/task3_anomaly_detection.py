#!/usr/bin/env python3
"""
任务3：真正的Isolation Forest机器学习实现（简化版）
不依赖pandas，直接用CSV和numpy
"""

import csv
import numpy as np
from collections import Counter, defaultdict

print("=" * 80)
print("任务3：Isolation Forest机器学习异常检测")
print("=" * 80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("正在加载数据...")
import os
# 获取脚本所在目录的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_file = os.path.join(parent_dir, 'data', 'task3_oncology_drug_event_pairs.csv')

data = []
with open(data_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"✓ 加载了 {len(data)} 条记录\n")

# ============================================================================
# 2. 特征工程
# ============================================================================

print("正在进行特征工程...")

drug_event_features = defaultdict(lambda: {
    'count': 0,
    'serious_count': 0,
    'death_count': 0,
    'hosp_count': 0
})

drug_totals = Counter()
event_totals = Counter()
total_reports = len(data)

for row in data:
    drug = row['target_drug']
    event = row['adverse_event']
    pair = f"{drug}||{event}"
    
    drug_totals[drug] += 1
    event_totals[event] += 1
    drug_event_features[pair]['count'] += 1
    
    try:
        if str(row.get('is_serious', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['serious_count'] += 1
        if str(row.get('is_death', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['death_count'] += 1
        if str(row.get('is_hospitalization', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['hosp_count'] += 1
    except:
        pass

print(f"✓ 识别了 {len(drug_event_features)} 个唯一的药物-事件对\n")

# ============================================================================
# 3. 构建特征矩阵
# ============================================================================

print("正在构建特征矩阵...")

feature_data = []
pair_names = []

for pair, stats in drug_event_features.items():
    drug, event = pair.split('||')
    
    count = stats['count']
    drug_total = drug_totals[drug]
    event_total = event_totals[event]
    
    # PRR
    a = count
    b = drug_total - count
    c = event_total - count
    d = total_reports - drug_total - event_total + count
    
    prr = 0
    if b > 0 and c > 0 and d > 0:
        prr = (a / b) / (c / d) if (c / d) > 0 else 0
    
    # ROR
    ror = 0
    if b > 0 and c > 0 and d > 0:
        ror = (a * d) / (b * c) if (b * c) > 0 else 0
    
    # 卡方
    expected_a = (drug_total * event_total) / total_reports
    chi2 = ((a - expected_a) ** 2) / expected_a if expected_a > 0 else 0
    
    # 严重性
    serious_rate = stats['serious_count'] / count if count > 0 else 0
    death_rate = stats['death_count'] / count if count > 0 else 0
    hosp_rate = stats['hosp_count'] / count if count > 0 else 0
    
    # 频率
    report_freq = count / total_reports
    
    features = [
        count, prr, ror, chi2, serious_rate, death_rate, 
        hosp_rate, report_freq, np.log(count + 1)
    ]
    
    feature_data.append(features)
    pair_names.append(pair)

X = np.array(feature_data)
print(f"✓ 特征矩阵形状: {X.shape}\n")

# ============================================================================
# 4. 简化的Isolation Forest实现
# ============================================================================

print("正在训练Isolation Forest模型...")

# 标准化
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练模型
iso_forest = IsolationForest(
    contamination=0.15,
    random_state=42,
    n_estimators=100,
    n_jobs=-1
)

predictions = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

n_anomalies = np.sum(predictions == -1)
print(f"✓ 模型训练完成")
print(f"✓ 检测到异常: {n_anomalies} 个\n")

# ============================================================================
# 5. 保存结果
# ============================================================================

print("正在保存结果...")

output_file = os.path.join(parent_dir, 'data', 'task3_anomalies_detected.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['drug', 'event', 'is_anomaly', 'anomaly_score', 'count', 'prr', 'death_rate'])
    
    for i, pair in enumerate(pair_names):
        drug, event = pair.split('||')
        writer.writerow([
            drug, event,
            'Yes' if predictions[i] == -1 else 'No',
            f"{-anomaly_scores[i]:.4f}",
            int(X[i, 0]),
            f"{X[i, 1]:.3f}",
            f"{X[i, 5]*100:.1f}%"
        ])

print(f"✓ 结果已保存到: {output_file}\n")

# ============================================================================
# 6. 展示Top异常
# ============================================================================

print("=" * 80)
print("Top 20 最显著的异常")
print("=" * 80)
print()

# 获取异常索引并按分数排序
anomaly_indices = np.where(predictions == -1)[0]
anomaly_indices_sorted = anomaly_indices[np.argsort(-anomaly_scores[anomaly_indices])]

for rank, idx in enumerate(anomaly_indices_sorted[:20], 1):
    drug, event = pair_names[idx].split('||')
    print(f"[{rank}] {drug} → {event}")
    print(f"    异常分数: {-anomaly_scores[idx]:.4f}")
    print(f"    PRR: {X[idx, 1]:.2f} | 死亡率: {X[idx, 5]*100:.1f}%")
    print()

print("=" * 80)
print("✅ 任务3完成！真正的Isolation Forest模型已训练。")
print("=" * 80)

